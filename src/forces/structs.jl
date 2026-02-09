using KernelAbstractions

mutable struct ForceData
    Fx::Float64; Fy::Float64; Fz::Float64
    
    Mx::Float64; My::Float64; Mz::Float64
    
    Fx_pressure::Float64; Fy_pressure::Float64; Fz_pressure::Float64
    Fx_viscous::Float64; Fy_viscous::Float64; Fz_viscous::Float64
    
    Cd::Float64
    Cl::Float64
    Cs::Float64
    Cmx::Float64
    Cmy::Float64
    Cmz::Float64
    
    rho_ref::Float64
    u_ref::Float64
    area_ref::Float64
    chord_ref::Float64
    moment_center::Tuple{Float64, Float64, Float64}
    
    force_scale::Float64
    length_scale::Float64
    
    symmetric::Bool
    
    # Plus-side stress maps (along +normal)
    pressure_map::AbstractArray
    shear_x_map::AbstractArray
    shear_y_map::AbstractArray
    shear_z_map::AbstractArray
    
    # Minus-side stress maps (along −normal)
    pressure_map_minus::AbstractArray
    shear_x_map_minus::AbstractArray
    shear_y_map_minus::AbstractArray
    shear_z_map_minus::AbstractArray
    
    diagnostics_printed::Bool
end

function ForceData(n_triangles::Int, backend; 
                   rho_ref=1.225, u_ref=10.0, area_ref=1.0, chord_ref=1.0,
                   moment_center=(0.0, 0.0, 0.0), force_scale=1.0, length_scale=1.0,
                   symmetric=false)
    
    p_map = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sx_map = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sy_map = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sz_map = KernelAbstractions.zeros(backend, Float32, n_triangles)

    p_map_minus = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sx_map_minus = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sy_map_minus = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sz_map_minus = KernelAbstractions.zeros(backend, Float32, n_triangles)

    return ForceData(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        rho_ref, u_ref, area_ref, chord_ref, 
        moment_center,
        force_scale, length_scale,
        symmetric,
        p_map, sx_map, sy_map, sz_map,
        p_map_minus, sx_map_minus, sy_map_minus, sz_map_minus,
        false
    )
end

function reset_forces!(fd::ForceData)
    fd.Fx = 0.0; fd.Fy = 0.0; fd.Fz = 0.0
    fd.Mx = 0.0; fd.My = 0.0; fd.Mz = 0.0
    fd.Fx_pressure = 0.0; fd.Fy_pressure = 0.0; fd.Fz_pressure = 0.0
    fd.Fx_viscous = 0.0; fd.Fy_viscous = 0.0; fd.Fz_viscous = 0.0
    fd.Cd = 0.0; fd.Cl = 0.0; fd.Cs = 0.0
    fd.Cmx = 0.0; fd.Cmy = 0.0; fd.Cmz = 0.0
end