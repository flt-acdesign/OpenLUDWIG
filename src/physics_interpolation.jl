// # FILE: .\src\physics_interpolation.jl
using KernelAbstractions

"""
PHYSICS_INTERPOLATION.JL - Multi-grid Interface Interpolation
"""

"""
Trilinear interpolation from coarse to fine grid with:
1. Temporal blending between old and new coarse data
2. f_neq rescaling to maintain stress continuity across levels

The f_neq rescaling is critical: f_neq encodes viscous stress, and since
ν = (τ-0.5)/3, we need f_neq_fine = f_neq_coarse × (τ_fine-0.5)/(τ_coarse-0.5)
"""
@inline function interpolate_with_rescaling(
    parent_f_new, parent_rho_new, parent_vel_new,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    k::Int32, block_size::Int32,
    w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
    tau_coarse::Float32, tau_fine::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    
    px_cont = (Float32(fine_gx) - 0.5f0) * 0.5f0
    py_cont = (Float32(fine_gy) - 0.5f0) * 0.5f0
    pz_cont = (Float32(fine_gz) - 0.5f0) * 0.5f0
    
    px0 = floor(Int32, px_cont)
    py0 = floor(Int32, py_cont)
    pz0 = floor(Int32, pz_cont)
    px1 = px0 + Int32(1)
    py1 = py0 + Int32(1)
    pz1 = pz0 + Int32(1)
    
    wx = px_cont - Float32(px0)
    wy = py_cont - Float32(py0)
    wz = pz_cont - Float32(pz0)
    
    px0 = max(Int32(1), px0)
    py0 = max(Int32(1), py0)
    pz0 = max(Int32(1), pz0)
    
    
    @inline function get_blended(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
        
        if pbx >= Int32(1) && pbx <= parent_dim_x && 
           pby >= Int32(1) && pby <= parent_dim_y && 
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                
                f_new = parent_f_new[plx, ply, plz, pb_idx, k]
                rho_new = parent_rho_new[plx, ply, plz, pb_idx]
                ux_new = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_new = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_new = parent_vel_new[plx, ply, plz, pb_idx, 3]
                
                if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0
                    f_old = parent_f_old[plx, ply, plz, pb_idx, k]
                    rho_old = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old = parent_vel_old[plx, ply, plz, pb_idx, 3]
                    
                    tw = temporal_weight
                    return (f_old*(1.0f0-tw) + f_new*tw,
                            rho_old*(1.0f0-tw) + rho_new*tw,
                            ux_old*(1.0f0-tw) + ux_new*tw,
                            uy_old*(1.0f0-tw) + uy_new*tw,
                            uz_old*(1.0f0-tw) + uz_new*tw, true)
                end
                return (f_new, rho_new, ux_new, uy_new, uz_new, true)
            end
        end
        return (w_k, 1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end
    
    
    d000 = get_blended(px0, py0, pz0)
    d100 = get_blended(px1, py0, pz0)
    d010 = get_blended(px0, py1, pz0)
    d110 = get_blended(px1, py1, pz0)
    d001 = get_blended(px0, py0, pz1)
    d101 = get_blended(px1, py0, pz1)
    d011 = get_blended(px0, py1, pz1)
    d111 = get_blended(px1, py1, pz1)
    
    
    v000 = d000
    v100 = d100[6] ? d100 : v000
    v010 = d010[6] ? d010 : v000
    v110 = d110[6] ? d110 : v000
    v001 = d001[6] ? d001 : v000
    v101 = d101[6] ? d101 : v000
    v011 = d011[6] ? d011 : v000
    v111 = d111[6] ? d111 : v000
    
    
    @inline function trilin(i)
        c00 = v000[i]*(1.0f0-wx) + v100[i]*wx
        c01 = v001[i]*(1.0f0-wx) + v101[i]*wx
        c10 = v010[i]*(1.0f0-wx) + v110[i]*wx
        c11 = v011[i]*(1.0f0-wx) + v111[i]*wx
        c0 = c00*(1.0f0-wy) + c10*wy
        c1 = c01*(1.0f0-wy) + c11*wy
        return c0*(1.0f0-wz) + c1*wz
    end
    
    f_int = trilin(1)
    rho_int = trilin(2)
    ux_int = trilin(3)
    uy_int = trilin(4)
    uz_int = trilin(5)
    
    
    feq_int = calculate_equilibrium(rho_int, ux_int, uy_int, uz_int, w_k, cx, cy, cz)
    f_neq = f_int - feq_int
    
    
    
    
    tau_c = tau_coarse - 0.5f0
    tau_f = tau_fine - 0.5f0
    scale = tau_c > 1.0f-6 ? clamp(tau_f / tau_c, 0.01f0, 100.0f0) : 1.0f0
    
    return feq_int + f_neq * scale
end