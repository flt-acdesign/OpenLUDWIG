# COEFFICIENTS.JL - Aerodynamic Force and Coefficient Computation
#
# Integrates surface pressure and shear stresses to compute:
# - Forces: Fx, Fy, Fz (drag, lift, side force in body axes)
# - Moments: Mx, My, Mz about a reference point
# - Coefficients: Cd, Cl, Cs, Cmx, Cmy, Cmz
#
# The integration is performed on GPU for efficiency.

using Printf
using Atomix

"""
GPU kernel to integrate forces and moments over all triangles.
Uses atomic operations to accumulate results from all threads.

Force from pressure: F_p = -∫ p * n dA  (pressure acts inward, opposite to outward normal)
Force from shear:    F_τ = ∫ τ dA       (shear acts tangentially)

Moment about reference point: M = ∫ r × (F_p + F_τ) dA
where r = triangle_center - reference_point
"""
@kernel function integrate_forces_kernel!(Fx_acc, Fy_acc, Fz_acc,
                                          Mx_acc, My_acc, Mz_acc,
                                          p_map, sx_map, sy_map, sz_map,
                                          cx_tri, cy_tri, cz_tri,
                                          nx_tri, ny_tri, nz_tri,
                                          areas,
                                          ref_x::Float32, ref_y::Float32, ref_z::Float32)
    i = @index(Global)
    @inbounds begin
        # Get triangle data
        p = p_map[i]
        tau_x = sx_map[i]
        tau_y = sy_map[i]
        tau_z = sz_map[i]
        
        nx = nx_tri[i]
        ny = ny_tri[i]
        nz = nz_tri[i]
        
        A = areas[i]
        
        # Force from pressure (acts opposite to outward normal)
        Fp_x = -p * nx * A
        Fp_y = -p * ny * A
        Fp_z = -p * nz * A
        
        # Force from shear (already in correct direction)
        Fs_x = tau_x * A
        Fs_y = tau_y * A
        Fs_z = tau_z * A
        
        # Total force on this triangle
        dFx = Fp_x + Fs_x
        dFy = Fp_y + Fs_y
        dFz = Fp_z + Fs_z
        
        # Moment arm from reference point to triangle center
        rx = cx_tri[i] - ref_x
        ry = cy_tri[i] - ref_y
        rz = cz_tri[i] - ref_z
        
        # Moment contribution: M = r × F
        dMx = ry * dFz - rz * dFy
        dMy = rz * dFx - rx * dFz
        dMz = rx * dFy - ry * dFx
        
        # Atomic accumulation
        Atomix.@atomic Fx_acc[1] += dFx
        Atomix.@atomic Fy_acc[1] += dFy
        Atomix.@atomic Fz_acc[1] += dFz
        Atomix.@atomic Mx_acc[1] += dMx
        Atomix.@atomic My_acc[1] += dMy
        Atomix.@atomic Mz_acc[1] += dMz
    end
end

"""
Alternative kernel that computes per-triangle contributions without atomics.
Use this with a reduction step for potentially better performance on large meshes.
"""
@kernel function compute_triangle_forces!(dFx, dFy, dFz, dMx, dMy, dMz,
                                          p_map, sx_map, sy_map, sz_map,
                                          cx_tri, cy_tri, cz_tri,
                                          nx_tri, ny_tri, nz_tri,
                                          areas,
                                          ref_x::Float32, ref_y::Float32, ref_z::Float32)
    i = @index(Global)
    @inbounds begin
        p = p_map[i]
        tau_x = sx_map[i]
        tau_y = sy_map[i]
        tau_z = sz_map[i]
        
        nx = nx_tri[i]
        ny = ny_tri[i]
        nz = nz_tri[i]
        
        A = areas[i]
        
        # Forces
        Fp_x = -p * nx * A
        Fp_y = -p * ny * A
        Fp_z = -p * nz * A
        
        Fs_x = tau_x * A
        Fs_y = tau_y * A
        Fs_z = tau_z * A
        
        fx = Fp_x + Fs_x
        fy = Fp_y + Fs_y
        fz = Fp_z + Fs_z
        
        # Moment arm
        rx = cx_tri[i] - ref_x
        ry = cy_tri[i] - ref_y
        rz = cz_tri[i] - ref_z
        
        # Store per-triangle values
        dFx[i] = fx
        dFy[i] = fy
        dFz[i] = fz
        dMx[i] = ry * fz - rz * fy
        dMy[i] = rz * fx - rx * fz
        dMz[i] = rx * fy - ry * fx
    end
end

"""
    AeroCoefficients

Stores aerodynamic forces, moments, and their non-dimensional coefficients.
"""
mutable struct AeroCoefficients
    # Forces in Newtons (body axes: x=streamwise, y=vertical, z=spanwise)
    Fx::Float64  # Drag direction (streamwise)
    Fy::Float64  # Lift direction (vertical)
    Fz::Float64  # Side force direction (spanwise)
    
    # Pressure and viscous components
    Fx_pressure::Float64
    Fy_pressure::Float64
    Fz_pressure::Float64
    Fx_viscous::Float64
    Fy_viscous::Float64
    Fz_viscous::Float64
    
    # Moments in N·m about reference point
    Mx::Float64  # Roll moment
    My::Float64  # Pitch moment
    Mz::Float64  # Yaw moment
    
    # Non-dimensional coefficients
    Cd::Float64  # Drag coefficient
    Cl::Float64  # Lift coefficient
    Cs::Float64  # Side force coefficient
    Cmx::Float64 # Roll moment coefficient
    Cmy::Float64 # Pitch moment coefficient
    Cmz::Float64 # Yaw moment coefficient
    
    # Reference values used for non-dimensionalization
    rho_ref::Float64      # Reference density [kg/m³]
    U_ref::Float64        # Reference velocity [m/s]
    A_ref::Float64        # Reference area [m²]
    L_ref::Float64        # Reference length for moments [m]
    q_inf::Float64        # Dynamic pressure = 0.5 * rho * U²
    
    # Reference point for moments
    ref_point::Tuple{Float64,Float64,Float64}
end

"""
    AeroCoefficients(; rho_ref, U_ref, A_ref, L_ref, ref_point)

Create an AeroCoefficients struct with specified reference values.

Arguments:
- rho_ref: Reference density [kg/m³] (typically freestream density)
- U_ref: Reference velocity [m/s] (typically freestream velocity)
- A_ref: Reference area [m²] (typically frontal area for drag, planform for lift)
- L_ref: Reference length [m] for moment coefficients (typically chord or diameter)
- ref_point: (x, y, z) reference point for moment computation [m]
"""
function AeroCoefficients(; rho_ref::Float64, U_ref::Float64, A_ref::Float64, 
                          L_ref::Float64, ref_point::Tuple{Float64,Float64,Float64})
    q_inf = 0.5 * rho_ref * U_ref^2
    return AeroCoefficients(
        0.0, 0.0, 0.0,  # Forces
        0.0, 0.0, 0.0,  # Pressure forces
        0.0, 0.0, 0.0,  # Viscous forces
        0.0, 0.0, 0.0,  # Moments
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Coefficients
        rho_ref, U_ref, A_ref, L_ref, q_inf,
        ref_point
    )
end

"""
    compute_aero_coefficients!(aero, force_data, gpu_mesh, backend; use_symmetry=false)

Compute aerodynamic forces, moments, and coefficients from surface stress data.

Arguments:
- aero: AeroCoefficients struct to store results
- force_data: ForceData struct with pressure and shear maps
- gpu_mesh: GPU mesh with triangle geometry
- backend: KernelAbstractions backend

Keyword Arguments:
- use_symmetry: If true, assume XZ symmetry plane and double Fx, Fz, My (set Fy, Mx, Mz to zero)
"""
function compute_aero_coefficients!(aero::AeroCoefficients, force_data::ForceData, 
                                    gpu_mesh, backend; use_symmetry::Bool=false)
    
    n_tri = gpu_mesh.n_triangles
    
    # Allocate GPU accumulators (single-element arrays for atomic ops)
    Fx_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fy_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fz_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mx_acc = KernelAbstractions.zeros(backend, Float32, 1)
    My_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mz_acc = KernelAbstractions.zeros(backend, Float32, 1)
    
    # Reference point
    ref_x = Float32(aero.ref_point[1])
    ref_y = Float32(aero.ref_point[2])
    ref_z = Float32(aero.ref_point[3])
    
    # Launch integration kernel
    kernel! = integrate_forces_kernel!(backend)
    kernel!(Fx_acc, Fy_acc, Fz_acc, Mx_acc, My_acc, Mz_acc,
            force_data.pressure_map, 
            force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
            gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
            gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
            gpu_mesh.areas,
            ref_x, ref_y, ref_z,
            ndrange=(n_tri,))
    
    KernelAbstractions.synchronize(backend)
    
    # Transfer results to CPU
    Fx = Float64(Array(Fx_acc)[1])
    Fy = Float64(Array(Fy_acc)[1])
    Fz = Float64(Array(Fz_acc)[1])
    Mx = Float64(Array(Mx_acc)[1])
    My = Float64(Array(My_acc)[1])
    Mz = Float64(Array(Mz_acc)[1])
    
    # Apply symmetry if requested (XZ plane symmetry)
    if use_symmetry
        Fx *= 2.0
        Fz *= 2.0
        My *= 2.0
        Fy = 0.0
        Mx = 0.0
        Mz = 0.0
    end
    
    # Store forces and moments
    aero.Fx = Fx
    aero.Fy = Fy
    aero.Fz = Fz
    aero.Mx = Mx
    aero.My = My
    aero.Mz = Mz
    
    # Compute coefficients
    q_A = aero.q_inf * aero.A_ref
    q_A_L = q_A * aero.L_ref
    
    if q_A > 1e-10
        aero.Cd = Fx / q_A    # Drag coefficient (force in x direction)
        aero.Cl = Fy / q_A    # Lift coefficient (force in y direction)  
        aero.Cs = Fz / q_A    # Side force coefficient (force in z direction)
    end
    
    if q_A_L > 1e-10
        aero.Cmx = Mx / q_A_L  # Roll moment coefficient
        aero.Cmy = My / q_A_L  # Pitch moment coefficient
        aero.Cmz = Mz / q_A_L  # Yaw moment coefficient
    end
    
    return aero
end

"""
    compute_aero_coefficients_decomposed!(aero, force_data, gpu_mesh, backend; use_symmetry=false)

Same as compute_aero_coefficients! but also computes pressure and viscous force components separately.
"""
function compute_aero_coefficients_decomposed!(aero::AeroCoefficients, force_data::ForceData,
                                               gpu_mesh, backend; use_symmetry::Bool=false)
    
    n_tri = gpu_mesh.n_triangles
    
    # Allocate per-triangle force arrays for reduction
    dFx = KernelAbstractions.zeros(backend, Float32, n_tri)
    dFy = KernelAbstractions.zeros(backend, Float32, n_tri)
    dFz = KernelAbstractions.zeros(backend, Float32, n_tri)
    dMx = KernelAbstractions.zeros(backend, Float32, n_tri)
    dMy = KernelAbstractions.zeros(backend, Float32, n_tri)
    dMz = KernelAbstractions.zeros(backend, Float32, n_tri)
    
    ref_x = Float32(aero.ref_point[1])
    ref_y = Float32(aero.ref_point[2])
    ref_z = Float32(aero.ref_point[3])
    
    # Compute per-triangle forces
    kernel! = compute_triangle_forces!(backend)
    kernel!(dFx, dFy, dFz, dMx, dMy, dMz,
            force_data.pressure_map,
            force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
            gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
            gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
            gpu_mesh.areas,
            ref_x, ref_y, ref_z,
            ndrange=(n_tri,))
    
    KernelAbstractions.synchronize(backend)
    
    # Transfer to CPU and sum
    dFx_cpu = Array(dFx)
    dFy_cpu = Array(dFy)
    dFz_cpu = Array(dFz)
    dMx_cpu = Array(dMx)
    dMy_cpu = Array(dMy)
    dMz_cpu = Array(dMz)
    
    Fx = Float64(sum(dFx_cpu))
    Fy = Float64(sum(dFy_cpu))
    Fz = Float64(sum(dFz_cpu))
    Mx = Float64(sum(dMx_cpu))
    My = Float64(sum(dMy_cpu))
    Mz = Float64(sum(dMz_cpu))
    
    # Compute pressure-only forces
    p_cpu = Array(force_data.pressure_map)
    nx_cpu = Array(gpu_mesh.normals_x)
    ny_cpu = Array(gpu_mesh.normals_y)
    nz_cpu = Array(gpu_mesh.normals_z)
    A_cpu = Array(gpu_mesh.areas)
    
    Fx_p = -sum(p_cpu .* nx_cpu .* A_cpu)
    Fy_p = -sum(p_cpu .* ny_cpu .* A_cpu)
    Fz_p = -sum(p_cpu .* nz_cpu .* A_cpu)
    
    # Viscous forces = total - pressure
    Fx_v = Fx - Fx_p
    Fy_v = Fy - Fy_p
    Fz_v = Fz - Fz_p
    
    # Apply symmetry
    if use_symmetry
        Fx *= 2.0; Fz *= 2.0; My *= 2.0
        Fx_p *= 2.0; Fz_p *= 2.0
        Fx_v *= 2.0; Fz_v *= 2.0
        Fy = 0.0; Mx = 0.0; Mz = 0.0
        Fy_p = 0.0; Fy_v = 0.0
    end
    
    # Store results
    aero.Fx = Fx
    aero.Fy = Fy
    aero.Fz = Fz
    aero.Fx_pressure = Fx_p
    aero.Fy_pressure = Fy_p
    aero.Fz_pressure = Fz_p
    aero.Fx_viscous = Fx_v
    aero.Fy_viscous = Fy_v
    aero.Fz_viscous = Fz_v
    aero.Mx = Mx
    aero.My = My
    aero.Mz = Mz
    
    # Compute coefficients
    q_A = aero.q_inf * aero.A_ref
    q_A_L = q_A * aero.L_ref
    
    if q_A > 1e-10
        aero.Cd = Fx / q_A
        aero.Cl = Fy / q_A
        aero.Cs = Fz / q_A
    end
    
    if q_A_L > 1e-10
        aero.Cmx = Mx / q_A_L
        aero.Cmy = My / q_A_L
        aero.Cmz = Mz / q_A_L
    end
    
    return aero
end

"""
    print_aero_summary(aero; show_decomposition=false)

Print a formatted summary of aerodynamic coefficients.
"""
function print_aero_summary(aero::AeroCoefficients; show_decomposition::Bool=false)
    println("\n" * "="^60)
    println("         AERODYNAMIC COEFFICIENTS SUMMARY")
    println("="^60)
    
    println("\nReference Values:")
    @printf("  ρ_ref  = %.4f kg/m³\n", aero.rho_ref)
    @printf("  U_ref  = %.4f m/s\n", aero.U_ref)
    @printf("  A_ref  = %.6f m²\n", aero.A_ref)
    @printf("  L_ref  = %.6f m\n", aero.L_ref)
    @printf("  q_∞    = %.4f Pa\n", aero.q_inf)
    @printf("  Ref point: (%.4f, %.4f, %.4f) m\n", aero.ref_point...)
    
    println("\nForces [N]:")
    @printf("  Fx (drag dir)  = %+.6e N\n", aero.Fx)
    @printf("  Fy (lift dir)  = %+.6e N\n", aero.Fy)
    @printf("  Fz (side dir)  = %+.6e N\n", aero.Fz)
    
    if show_decomposition
        println("\n  Pressure contribution:")
        @printf("    Fx_p = %+.6e N\n", aero.Fx_pressure)
        @printf("    Fy_p = %+.6e N\n", aero.Fy_pressure)
        @printf("    Fz_p = %+.6e N\n", aero.Fz_pressure)
        println("  Viscous contribution:")
        @printf("    Fx_v = %+.6e N\n", aero.Fx_viscous)
        @printf("    Fy_v = %+.6e N\n", aero.Fy_viscous)
        @printf("    Fz_v = %+.6e N\n", aero.Fz_viscous)
    end
    
    println("\nMoments [N·m] about reference point:")
    @printf("  Mx (roll)  = %+.6e N·m\n", aero.Mx)
    @printf("  My (pitch) = %+.6e N·m\n", aero.My)
    @printf("  Mz (yaw)   = %+.6e N·m\n", aero.Mz)
    
    println("\nCoefficients:")
    @printf("  Cd = %+.6f\n", aero.Cd)
    @printf("  Cl = %+.6f\n", aero.Cl)
    @printf("  Cs = %+.6f\n", aero.Cs)
    @printf("  Cmx = %+.6f\n", aero.Cmx)
    @printf("  Cmy = %+.6f\n", aero.Cmy)
    @printf("  Cmz = %+.6f\n", aero.Cmz)
    
    println("="^60 * "\n")
end

"""
    write_aero_csv_header(filepath)

Write CSV header for time-history of aerodynamic coefficients.
"""
function write_aero_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io, "step,time_s,Fx_N,Fy_N,Fz_N,Mx_Nm,My_Nm,Mz_Nm,Cd,Cl,Cs,Cmx,Cmy,Cmz")
    end
end

"""
    append_aero_csv(filepath, step, time, aero)

Append one row of aerodynamic data to CSV file.
"""
function append_aero_csv(filepath::String, step::Int, time::Float64, aero::AeroCoefficients)
    open(filepath, "a") do io
        @printf(io, "%d,%.6e,%+.6e,%+.6e,%+.6e,%+.6e,%+.6e,%+.6e,%+.6f,%+.6f,%+.6f,%+.6f,%+.6f,%+.6f\n",
                step, time,
                aero.Fx, aero.Fy, aero.Fz,
                aero.Mx, aero.My, aero.Mz,
                aero.Cd, aero.Cl, aero.Cs,
                aero.Cmx, aero.Cmy, aero.Cmz)
    end
end
