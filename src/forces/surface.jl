"""
SURFACE.JL - Surface Stress Mapping and Force Integration

This module:
1. Maps LBM flow data (density, velocity) to surface pressure and shear stress
2. Integrates surface stresses to compute total aerodynamic forces and moments
3. Computes aerodynamic coefficients (Cd, Cl, Cs, Cm)

Coordinate convention (standard aircraft):
- X: Streamwise (flow direction) → Drag
- Y: Spanwise → Side force  
- Z: Vertical → Lift
"""

using Printf
using KernelAbstractions
using Atomix

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Compute pressure and wall shear stress from near-wall fluid cell data.

Physics:
- Pressure: p = (ρ - ρ₀) * cs² in lattice units, then scaled to physical
- Wall shear: τ_w = μ * (u_tangential / distance_to_wall)

Returns: (pressure_Pa, tau_x_Pa, tau_y_Pa, tau_z_Pa)
"""
@inline function compute_stress_from_cell(
    rho_val::Float32, 
    ux::Float32, uy::Float32, uz::Float32,
    nx::Float32, ny::Float32, nz::Float32,  # Surface normal (outward)
    dist_to_wall::Float32,                   # Distance from cell center to wall [lattice units]
    tau_molecular::Float32,                  # LBM relaxation time
    pressure_scale::Float32,                 # Conversion factor for pressure
    stress_scale::Float32                    # Conversion factor for stress
)
    # -------------------------------------------------------------------------
    # PRESSURE
    # -------------------------------------------------------------------------
    # In LBM: p_lat = ρ * cs² where cs² = 1/3
    # Gauge pressure (deviation from reference ρ=1): p_gauge_lat = (ρ - 1) / 3
    # Physical pressure: p_phys = p_gauge_lat * pressure_scale
    
    p_gauge_lat = (rho_val - 1.0f0) / 3.0f0
    p_phys = p_gauge_lat * pressure_scale
    
    # -------------------------------------------------------------------------
    # WALL SHEAR STRESS
    # -------------------------------------------------------------------------
    # Shear stress acts tangentially to the surface in the direction of flow
    # τ_w = μ * du/dn ≈ ρ * ν * u_tangential / distance
    
    # Compute tangential velocity (remove normal component)
    u_dot_n = ux * nx + uy * ny + uz * nz
    ut_x = ux - u_dot_n * nx
    ut_y = uy - u_dot_n * ny
    ut_z = uz - u_dot_n * nz
    
    # Tangential velocity magnitude
    u_tan_mag = sqrt(ut_x * ut_x + ut_y * ut_y + ut_z * ut_z)
    
    # Kinematic viscosity in lattice units: ν = (τ - 0.5) / 3
    nu_lat = (tau_molecular - 0.5f0) / 3.0f0
    
    # Initialize shear stress components
    tau_x = 0.0f0
    tau_y = 0.0f0
    tau_z = 0.0f0
    
    if u_tan_mag > 1.0f-10 && dist_to_wall > 0.01f0
        # Wall shear stress magnitude in lattice units
        # τ_lat = ρ * ν * u_tan / y  (where y is distance to wall)
        tau_lat_mag = rho_val * nu_lat * u_tan_mag / dist_to_wall
        
        # Convert to physical units and apply as vector in tangential direction
        tau_phys_mag = tau_lat_mag * stress_scale
        
        # Shear stress vector (in direction of tangential flow)
        tau_x = (ut_x / u_tan_mag) * tau_phys_mag
        tau_y = (ut_y / u_tan_mag) * tau_phys_mag
        tau_z = (ut_z / u_tan_mag) * tau_phys_mag
    end
    
    return (p_phys, tau_x, tau_y, tau_z)
end

"""
Get block index and local coordinates for a global grid position.
Returns (block_idx, local_x, local_y, local_z, is_valid)
"""
@inline function get_cell_at_position(
    gx::Int32, gy::Int32, gz::Int32,
    block_ptr, 
    block_size::Int32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32
)
    if gx < Int32(1) || gy < Int32(1) || gz < Int32(1)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end
    
    bs = block_size
    bx = (gx - Int32(1)) ÷ bs + Int32(1)
    by = (gy - Int32(1)) ÷ bs + Int32(1)
    bz = (gz - Int32(1)) ÷ bs + Int32(1)
    
    if bx < Int32(1) || bx > dim_x || by < Int32(1) || by > dim_y || bz < Int32(1) || bz > dim_z
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end
    
    b_idx = block_ptr[bx, by, bz]
    if b_idx <= Int32(0)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end
    
    lx = ((gx - Int32(1)) % bs) + Int32(1)
    ly = ((gy - Int32(1)) % bs) + Int32(1)
    lz = ((gz - Int32(1)) % bs) + Int32(1)
    
    return (b_idx, lx, ly, lz, true)
end

# =============================================================================
# SURFACE STRESS MAPPING KERNEL
# =============================================================================

"""
GPU kernel to map LBM data to surface triangles.

For each triangle:
1. Find the nearest fluid cell to the triangle center
2. Compute pressure and shear stress from that cell's data
3. Store results in the per-triangle stress maps
"""
@kernel function map_stresses_kernel!(
    p_map, sx_map, sy_map, sz_map,
    # Triangle geometry (in STL coordinates, before mesh offset)
    cx_tri, cy_tri, cz_tri,      # Triangle centers
    nx_tri, ny_tri, nz_tri,      # Triangle normals (outward)
    # LBM data
    rho, vel, obstacle, block_ptr,
    # Grid parameters
    block_size::Int32, dx::Float32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32,
    # Coordinate transformation
    mesh_offset_x::Float32, mesh_offset_y::Float32, mesh_offset_z::Float32,
    # Scaling factors
    pressure_scale::Float32, stress_scale::Float32,
    # LBM parameter
    tau_molecular::Float32,
    # Search parameters
    search_radius::Int32
)
    i = @index(Global)
    
    @inbounds begin
        # Triangle center in LBM coordinates (apply mesh offset)
        tx = cx_tri[i] + mesh_offset_x
        ty = cy_tri[i] + mesh_offset_y
        tz = cz_tri[i] + mesh_offset_z
        
        # Triangle outward normal
        n_x = nx_tri[i]
        n_y = ny_tri[i]
        n_z = nz_tri[i]
        
        # Convert to grid coordinates
        gx_f = tx / dx
        gy_f = ty / dx
        gz_f = tz / dx
        
        g_x = floor(Int32, gx_f) + Int32(1)
        g_y = floor(Int32, gy_f) + Int32(1)
        g_z = floor(Int32, gz_f) + Int32(1)
        
        bs = block_size
        
        # Search for nearest fluid cell
        best_dist_sq = Float32(1e10)
        best_rho = 1.0f0
        best_ux = 0.0f0
        best_uy = 0.0f0
        best_uz = 0.0f0
        best_wall_dist = 0.5f0  # Default: assume cell center is 0.5 dx from wall
        found_fluid = false
        
        # Search in expanding shells
        for radius in Int32(0):search_radius
            # Early exit if we found a fluid cell in previous shell
            if found_fluid && radius > Int32(1)
                break
            end
            
            for dz in -radius:radius
                for dy in -radius:radius
                    for ddx in -radius:radius
                        # Only check shell boundary (skip interior for radius > 0)
                        if radius > Int32(0)
                            at_shell = (abs(ddx) == radius) || (abs(dy) == radius) || (abs(dz) == radius)
                            if !at_shell
                                continue
                            end
                        end
                        
                        check_gx = g_x + Int32(ddx)
                        check_gy = g_y + Int32(dy)
                        check_gz = g_z + Int32(dz)
                        
                        b_idx, lx, ly, lz, valid = get_cell_at_position(
                            check_gx, check_gy, check_gz,
                            block_ptr, bs, dim_x, dim_y, dim_z
                        )
                        
                        # Check if this is a valid fluid cell (not obstacle)
                        if valid && !obstacle[lx, ly, lz, b_idx]
                            # Cell center in physical coordinates
                            cell_cx = (Float32(check_gx) - 0.5f0) * dx
                            cell_cy = (Float32(check_gy) - 0.5f0) * dx
                            cell_cz = (Float32(check_gz) - 0.5f0) * dx
                            
                            dist_sq = (tx - cell_cx)^2 + (ty - cell_cy)^2 + (tz - cell_cz)^2
                            
                            if dist_sq < best_dist_sq
                                best_dist_sq = dist_sq
                                best_rho = rho[lx, ly, lz, b_idx]
                                best_ux = vel[lx, ly, lz, b_idx, 1]
                                best_uy = vel[lx, ly, lz, b_idx, 2]
                                best_uz = vel[lx, ly, lz, b_idx, 3]
                                # Estimate wall distance: distance from cell to triangle
                                best_wall_dist = sqrt(dist_sq) / dx  # In lattice units
                                found_fluid = true
                            end
                        end
                    end
                end
            end
        end
        
        # Compute stresses if fluid cell found
        p_val = 0.0f0
        tau_x = 0.0f0
        tau_y = 0.0f0
        tau_z = 0.0f0
        
        if found_fluid
            # Ensure minimum wall distance for numerical stability
            wall_dist = max(best_wall_dist, 0.5f0)
            
            p_val, tau_x, tau_y, tau_z = compute_stress_from_cell(
                best_rho, best_ux, best_uy, best_uz,
                n_x, n_y, n_z,
                wall_dist, tau_molecular,
                pressure_scale, stress_scale
            )
        end
        
        # Store results
        p_map[i] = p_val
        sx_map[i] = tau_x
        sy_map[i] = tau_y
        sz_map[i] = tau_z
    end
end

# =============================================================================
# FORCE INTEGRATION KERNEL
# =============================================================================

"""
GPU kernel to integrate surface stresses into total forces and moments.

Force contributions:
- Pressure force: dF_p = -p * n * dA  (pressure acts inward, opposite to outward normal)
- Viscous force:  dF_v = τ * dA       (shear stress already in correct direction)

Moment about reference point:
- dM = r × dF  where r = triangle_center - reference_point
"""
@kernel function integrate_forces_kernel!(
    # Accumulators (atomic)
    Fx_p_acc, Fy_p_acc, Fz_p_acc,  # Pressure force components
    Fx_v_acc, Fy_v_acc, Fz_v_acc,  # Viscous force components
    Mx_acc, My_acc, Mz_acc,        # Moment components
    # Surface stress data
    p_map, sx_map, sy_map, sz_map,
    # Triangle geometry
    cx_tri, cy_tri, cz_tri,        # Centers (in simulation coordinates)
    nx_tri, ny_tri, nz_tri,        # Outward normals
    areas,                          # Triangle areas [m²]
    # Coordinate offset
    offset_x::Float32, offset_y::Float32, offset_z::Float32,
    # Moment reference point
    ref_x::Float32, ref_y::Float32, ref_z::Float32
)
    i = @index(Global)
    
    @inbounds begin
        # Surface stress values
        p = p_map[i]
        tau_x = sx_map[i]
        tau_y = sy_map[i]
        tau_z = sz_map[i]
        
        # Triangle geometry
        nx = nx_tri[i]
        ny = ny_tri[i]
        nz = nz_tri[i]
        A = areas[i]
        
        # Triangle center in simulation coordinates
        cx = cx_tri[i] + offset_x
        cy = cy_tri[i] + offset_y
        cz = cz_tri[i] + offset_z
        
        # ---------------------------------------------------------------------
        # FORCE COMPUTATION
        # ---------------------------------------------------------------------
        # Pressure force: F_p = -∫ p * n dA
        # The negative sign because pressure acts inward (opposite to outward normal)
        dFp_x = -p * nx * A
        dFp_y = -p * ny * A
        dFp_z = -p * nz * A
        
        # Viscous force: F_v = ∫ τ dA
        # τ is already the stress vector in the direction opposing the flow
        dFv_x = tau_x * A
        dFv_y = tau_y * A
        dFv_z = tau_z * A
        
        # Total force on this triangle
        dFx = dFp_x + dFv_x
        dFy = dFp_y + dFv_y
        dFz = dFp_z + dFv_z
        
        # ---------------------------------------------------------------------
        # MOMENT COMPUTATION
        # ---------------------------------------------------------------------
        # Position vector from reference point to triangle center
        rx = cx - ref_x
        ry = cy - ref_y
        rz = cz - ref_z
        
        # Moment: M = r × F
        dMx = ry * dFz - rz * dFy
        dMy = rz * dFx - rx * dFz
        dMz = rx * dFy - ry * dFx
        
        # ---------------------------------------------------------------------
        # ATOMIC ACCUMULATION
        # ---------------------------------------------------------------------
        Atomix.@atomic Fx_p_acc[1] += dFp_x
        Atomix.@atomic Fy_p_acc[1] += dFp_y
        Atomix.@atomic Fz_p_acc[1] += dFp_z
        
        Atomix.@atomic Fx_v_acc[1] += dFv_x
        Atomix.@atomic Fy_v_acc[1] += dFv_y
        Atomix.@atomic Fz_v_acc[1] += dFv_z
        
        Atomix.@atomic Mx_acc[1] += dMx
        Atomix.@atomic My_acc[1] += dMy
        Atomix.@atomic Mz_acc[1] += dMz
    end
end

# =============================================================================
# PUBLIC API
# =============================================================================

"""
    map_surface_stresses!(force_data, level, gpu_mesh, backend, params; search_radius=5)

Map LBM flow data to surface stress on each triangle.

This populates force_data.pressure_map and force_data.shear_x/y/z_map.

Arguments:
- force_data: ForceData struct to store results
- level: The finest grid level with Bouzidi boundaries
- gpu_mesh: GPU mesh with triangle centers and normals
- backend: KernelAbstractions backend
- params: Domain parameters (contains mesh_offset, time_scale, etc.)

Keyword Arguments:
- search_radius: Maximum cells to search for fluid data (default: 5)
"""
function map_surface_stresses!(force_data::ForceData, level, gpu_mesh,
                               backend, params; search_radius::Int=5)
    
    mesh_offset = params.mesh_offset
    tau_mol = level.tau
    
    # Compute scaling factors
    # pressure_scale: converts (ρ-1)/3 to Pascals
    # In LBM: p_lat = ρ * cs² = ρ/3
    # p_phys = p_lat * ρ_phys * (dx/dt)² = p_lat * ρ_phys * velocity_scale²
    velocity_scale = params.velocity_scale
    rho_phys = params.rho_physical
    
    pressure_scale = Float32(rho_phys * velocity_scale * velocity_scale)
    stress_scale = pressure_scale  # Same scaling for shear stress
    
    # Launch mapping kernel
    kernel! = map_stresses_kernel!(backend)
    kernel!(
        force_data.pressure_map, 
        force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
        gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        level.rho, level.vel, level.obstacle, level.block_pointer,
        Int32(BLOCK_SIZE), Float32(level.dx),
        Int32(level.grid_dim_x), Int32(level.grid_dim_y), Int32(level.grid_dim_z),
        Float32(mesh_offset[1]), Float32(mesh_offset[2]), Float32(mesh_offset[3]),
        pressure_scale, stress_scale,
        Float32(tau_mol),
        Int32(search_radius),
        ndrange=(gpu_mesh.n_triangles,)
    )
    
    KernelAbstractions.synchronize(backend)
    
    # Print diagnostics (once)
    if !force_data.diagnostics_printed
        p_cpu = Array(force_data.pressure_map)
        sx_cpu = Array(force_data.shear_x_map)
        
        n_total = gpu_mesh.n_triangles
        n_nonzero_p = count(x -> abs(x) > 1e-10, p_cpu)
        n_nonzero_s = count(x -> abs(x) > 1e-10, sx_cpu)
        
        println("[Surface] Mapped $(n_total) triangles:")
        println("          Mesh offset: ($(round(mesh_offset[1], digits=3)), $(round(mesh_offset[2], digits=3)), $(round(mesh_offset[3], digits=3)))")
        println("          Grid spacing dx = $(round(level.dx, digits=6)) m")
        println("          Pressure scale = $(round(pressure_scale, digits=2)) Pa")
        println("          Coverage: pressure=$(n_nonzero_p)/$(n_total) ($(round(100*n_nonzero_p/n_total, digits=1))%)")
        
        if n_nonzero_p > 0
            p_nz = filter(x -> abs(x) > 1e-10, p_cpu)
            @printf("          Pressure range: [%.3e, %.3e] Pa\n", minimum(p_nz), maximum(p_nz))
        end
        
        force_data.diagnostics_printed = true
    end
end

"""
    integrate_surface_forces!(force_data, gpu_mesh, backend, params)

Integrate surface stresses to compute total aerodynamic forces, moments, and coefficients.

This should be called AFTER map_surface_stresses!().

The function:
1. Integrates pressure and shear over all triangles → total force
2. Computes moments about the reference point
3. Applies symmetry correction if enabled
4. Computes non-dimensional coefficients (Cd, Cl, Cs, Cm)

Arguments:
- force_data: ForceData struct (must have stress maps populated)
- gpu_mesh: GPU mesh with triangle geometry
- backend: KernelAbstractions backend
- params: Domain parameters
"""
function integrate_surface_forces!(force_data::ForceData, gpu_mesh, backend, params)
    n_tri = gpu_mesh.n_triangles
    
    # Reset force accumulation
    reset_forces!(force_data)
    
    # Allocate atomic accumulators
    Fx_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fy_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fz_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fx_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fy_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fz_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mx_acc = KernelAbstractions.zeros(backend, Float32, 1)
    My_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mz_acc = KernelAbstractions.zeros(backend, Float32, 1)
    
    # Reference point for moments
    mc = force_data.moment_center
    mesh_offset = params.mesh_offset
    
    # Launch integration kernel
    kernel! = integrate_forces_kernel!(backend)
    kernel!(
        Fx_p_acc, Fy_p_acc, Fz_p_acc,
        Fx_v_acc, Fy_v_acc, Fz_v_acc,
        Mx_acc, My_acc, Mz_acc,
        force_data.pressure_map,
        force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
        gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        gpu_mesh.areas,
        Float32(mesh_offset[1]), Float32(mesh_offset[2]), Float32(mesh_offset[3]),
        Float32(mc[1]), Float32(mc[2]), Float32(mc[3]),
        ndrange=(n_tri,)
    )
    
    KernelAbstractions.synchronize(backend)
    
    # Transfer results from GPU
    Fx_p = Float64(Array(Fx_p_acc)[1])
    Fy_p = Float64(Array(Fy_p_acc)[1])
    Fz_p = Float64(Array(Fz_p_acc)[1])
    Fx_v = Float64(Array(Fx_v_acc)[1])
    Fy_v = Float64(Array(Fy_v_acc)[1])
    Fz_v = Float64(Array(Fz_v_acc)[1])
    Mx = Float64(Array(Mx_acc)[1])
    My = Float64(Array(My_acc)[1])
    Mz = Float64(Array(Mz_acc)[1])
    
    # Apply symmetry correction (if simulating half domain with Y=0 symmetry plane)
    if force_data.symmetric
        # Double the forces in X and Z (drag and lift)
        # Y force and X,Z moments should be zero by symmetry
        Fx_p *= 2.0; Fz_p *= 2.0
        Fx_v *= 2.0; Fz_v *= 2.0
        My *= 2.0  # Pitching moment doubles
        Fy_p = 0.0; Fy_v = 0.0  # Side force = 0
        Mx = 0.0; Mz = 0.0      # Roll and yaw moments = 0
    end
    
    # Store force components
    force_data.Fx_pressure = Fx_p
    force_data.Fy_pressure = Fy_p
    force_data.Fz_pressure = Fz_p
    force_data.Fx_viscous = Fx_v
    force_data.Fy_viscous = Fy_v
    force_data.Fz_viscous = Fz_v
    
    # Total forces
    force_data.Fx = Fx_p + Fx_v
    force_data.Fy = Fy_p + Fy_v
    force_data.Fz = Fz_p + Fz_v
    
    # Moments
    force_data.Mx = Mx
    force_data.My = My
    force_data.Mz = Mz
    
    # Compute aerodynamic coefficients
    # Dynamic pressure: q = 0.5 * ρ * U²
    q_inf = 0.5 * force_data.rho_ref * force_data.u_ref^2
    
    # Reference force: F_ref = q * A
    F_ref = q_inf * force_data.area_ref
    
    # Reference moment: M_ref = q * A * L
    M_ref = F_ref * force_data.chord_ref
    
    if F_ref > 1e-10
        # Cd = Drag / (q * A)    [X direction is flow/drag]
        force_data.Cd = force_data.Fx / F_ref
        
        # Cl = Lift / (q * A)    [Z direction is lift]
        force_data.Cl = force_data.Fz / F_ref
        
        # Cs = Side / (q * A)    [Y direction is side]
        force_data.Cs = force_data.Fy / F_ref
    end
    
    if M_ref > 1e-10
        force_data.Cmx = force_data.Mx / M_ref  # Roll moment coeff
        force_data.Cmy = force_data.My / M_ref  # Pitch moment coeff
        force_data.Cmz = force_data.Mz / M_ref  # Yaw moment coeff
    end
end

"""
    compute_aerodynamics!(force_data, level, gpu_mesh, backend, params; search_radius=5)

Complete aerodynamic computation: map stresses and integrate forces.

This is the main entry point for aerodynamic coefficient computation.
Call this each time you want updated force/coefficient values.

Arguments:
- force_data: ForceData struct to store all results
- level: Finest grid level with boundary data
- gpu_mesh: GPU mesh with triangle geometry
- backend: KernelAbstractions backend  
- params: Domain parameters

Keyword Arguments:
- search_radius: Cell search distance for stress mapping (default: 5)
"""
function compute_aerodynamics!(force_data::ForceData, level, gpu_mesh, backend, params;
                               search_radius::Int=5)
    # Step 1: Map flow data to surface stresses
    map_surface_stresses!(force_data, level, gpu_mesh, backend, params; 
                          search_radius=search_radius)
    
    # Step 2: Integrate stresses to get forces and coefficients
    integrate_surface_forces!(force_data, gpu_mesh, backend, params)
end