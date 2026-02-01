// # FILE: .\src\physics_kernels.jl
using KernelAbstractions
using StaticArrays

"""
PHYSICS_KERNELS.JL - Main LBM Stream-Collide Kernel
"""

@kernel function stream_collide_kernel_v2!(
    f_out, f_in, f_post_collision,
    rho_out, vel_out, vel_in,
    obstacle, sponge_arr, wall_dist_arr,
    neighbor_table,
    active_coords_x, active_coords_y, active_coords_z,
    parent_f, parent_rho, parent_vel, parent_ptr,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    tau_molecular::Float32,
    tau_parent::Float32,
    c_wale::Float32,
    nu_sgs_background::Float32,
    cs2_val::Float32, cs4_val::Float32,
    is_level_1::Int32,
    is_symmetric::Int32,
    nx_global::Int32, ny_global::Int32, nz_global::Int32,
    u_inlet::Float32,
    n_blocks::Int32,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr, opp_arr,
    mirror_y_arr, mirror_z_arr,
    wall_model_active::Int32,
    time_step_seed::Int32,
    store_post_collision::Int32,
    inlet_turbulence::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    sponge_blend_distributions::Int32
)
    x, y, z, b_idx = @index(Global, NTuple)
    
    if b_idx <= n_blocks
        @inbounds begin
            bx = active_coords_x[b_idx]
            by = active_coords_y[b_idx]
            bz = active_coords_z[b_idx]
            gx = (bx - Int32(1)) * block_size + Int32(x)
            gy = (by - Int32(1)) * block_size + Int32(y)
            gz = (bz - Int32(1)) * block_size + Int32(z)
            
            is_obs = obstacle[x, y, z, b_idx]
            
            rho = 0.0f0
            jx = 0.0f0
            jy = 0.0f0
            jz = 0.0f0
            f_stored = MVector{27, Float32}(undef)
            
            
            
            
            
            for k in Int32(1):Int32(27)
                cx = cx_arr[k]
                cy = cy_arr[k]
                cz = cz_arr[k]
                
                sx = Int32(x) - cx
                sy = Int32(y) - cy
                sz = Int32(z) - cz
                
                val = 0.0f0
                
                if sx >= Int32(1) && sx <= block_size && sy >= Int32(1) && sy <= block_size && sz >= Int32(1) && sz <= block_size
                    val = f_in[sx, sy, sz, b_idx, k]
                else
                    nb_off_x = sx < Int32(1) ? Int32(-1) : (sx > block_size ? Int32(1) : Int32(0))
                    nb_off_y = sy < Int32(1) ? Int32(-1) : (sy > block_size ? Int32(1) : Int32(0))
                    nb_off_z = sz < Int32(1) ? Int32(-1) : (sz > block_size ? Int32(1) : Int32(0))
                    dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1))*Int32(3) + (nb_off_z + Int32(1))*Int32(9) + Int32(1)
                    nb_idx_global = neighbor_table[b_idx, dir_idx]
                    
                    if nb_idx_global > Int32(0)
                        nsx = sx < Int32(1) ? sx + block_size : (sx > block_size ? sx - block_size : sx)
                        nsy = sy < Int32(1) ? sy + block_size : (sy > block_size ? sy - block_size : sy)
                        nsz = sz < Int32(1) ? sz + block_size : (sz > block_size ? sz - block_size : sz)
                        val = f_in[nsx, nsy, nsz, nb_idx_global, k]
                    else
                        src_gx = gx - cx
                        src_gy = gy - cy
                        src_gz = gz - cz
                        
                        is_inlet = src_gx < Int32(1)
                        is_outlet = src_gx > nx_global
                        is_y_min = src_gy < Int32(1)
                        is_y_max = src_gy > ny_global
                        is_z_min = src_gz < Int32(1)
                        is_z_max = src_gz > nz_global
                        
                        if is_inlet
                            
                            noise = inlet_turbulence > 0.0f0 ? gradient_noise(gy, gz, time_step_seed, Int32(1234)) * inlet_turbulence * u_inlet : 0.0f0
                            u_inst = u_inlet + noise
                            cu_in = Float32(cx) * u_inst
                            val = w_arr[k] * (1.0f0 + 3.0f0*cu_in + 4.5f0*cu_in*cu_in - 1.5f0*u_inst*u_inst)
                            
                        elseif is_outlet
                            
                            
                            
                            
                            
                            cu_out = Float32(cx) * u_inlet
                            val = w_arr[k] * (1.0f0 + 3.0f0*cu_out + 4.5f0*cu_out*cu_out - 1.5f0*u_inlet*u_inlet)
                            
                        elseif is_y_min && is_symmetric == Int32(1)
                            val = f_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif is_y_min || is_y_max
                            val = f_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif is_z_min || is_z_max
                            val = f_in[x, y, z, b_idx, mirror_z_arr[k]]
                            
                        elseif is_level_1 == Int32(0)
                            
                            
                            
                            val = interpolate_with_rescaling(
                                parent_f, parent_rho, parent_vel,
                                parent_f_old, parent_rho_old, parent_vel_old,
                                parent_ptr,
                                parent_dim_x, parent_dim_y, parent_dim_z,
                                src_gx, src_gy, src_gz,
                                k, block_size, w_arr[k],
                                Float32(cx), Float32(cy), Float32(cz),
                                tau_parent, tau_molecular,
                                temporal_weight,
                                use_temporal_interp
                            )
                        else
                            val = w_arr[k]
                        end
                    end
                end
                
                f_stored[k] = val
                rho += val
                jx += val * Float32(cx_arr[k])
                jy += val * Float32(cy_arr[k])
                jz += val * Float32(cz_arr[k])
            end

            
            
            
            if is_obs
                vel_out[x, y, z, b_idx, 1] = 0.0f0
                vel_out[x, y, z, b_idx, 2] = 0.0f0
                vel_out[x, y, z, b_idx, 3] = 0.0f0
                rho_out[x, y, z, b_idx] = 1.0f0
                
                for k in Int32(1):Int32(27)
                    f_coll = f_stored[opp_arr[k]]
                    f_out[x, y, z, b_idx, k] = f_coll
                    if store_post_collision == Int32(1)
                        f_post_collision[x, y, z, b_idx, k] = f_coll
                    end
                end
            else
                
                
                
                
                rho = max(rho, 0.01f0)
                inv_rho = 1.0f0 / rho
                ux = jx * inv_rho
                uy = jy * inv_rho
                uz = jz * inv_rho
                
                
                
                
                sp = sponge_arr[x, y, z, b_idx]
                if sp > 0.0f0
                    rho_target = 1.0f0
                    ux_target = u_inlet
                    
                    rho = rho * (1.0f0 - sp) + rho_target * sp
                    ux = ux * (1.0f0 - sp) + ux_target * sp
                    uy = uy * (1.0f0 - sp)
                    uz = uz * (1.0f0 - sp)
                    
                    
                    if sponge_blend_distributions == Int32(1)
                        for k in Int32(1):Int32(27)
                            feq_target = calculate_equilibrium(rho_target, ux_target, 0.0f0, 0.0f0,
                                                               w_arr[k], Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]))
                            f_stored[k] = f_stored[k] * (1.0f0 - sp) + feq_target * sp
                        end
                    end
                end
                
                
                Fx_wall = 0.0f0
                Fy_wall = 0.0f0
                Fz_wall = 0.0f0
                
                if wall_model_active == Int32(1)
                    dist_wall = wall_dist_arr[x, y, z, b_idx]
                    if dist_wall > 0.0f0 && dist_wall < 10.0f0
                        u_mag = sqrt(ux*ux + uy*uy + uz*uz)
                        nu_visc = (tau_molecular - 0.5f0) / 3.0f0
                        
                        if u_mag > 1.0f-6 && nu_visc > 1.0f-10
                            u_tau = u_mag * (nu_visc / (dist_wall * u_mag + 1.0f-10))^(1.0f0/7.0f0) * (2.0f0 * 8.3f0)^(-1.0f0/7.0f0)
                            u_tau = max(u_tau, 1.0f-6)
                            
                            y_p = u_tau * dist_wall / nu_visc
                            if y_p > 11.81f0
                                u_plus_law = (1.0f0 / KAPPA) * log(y_p) + 5.2f0
                                if u_plus_law > 0.1f0
                                    u_tau = u_tau * ((u_mag / u_tau) / u_plus_law)
                                    u_tau = max(u_tau, 1.0f-6)
                                end
                            end
                            
                            tau_wall = rho * u_tau * u_tau
                            tau_res = rho * nu_visc * (u_mag / dist_wall)
                            
                            if tau_wall > tau_res
                                force_mag = (tau_wall - tau_res) / dist_wall
                                Fx_wall = -force_mag * ux / u_mag
                                Fy_wall = -force_mag * uy / u_mag
                                Fz_wall = -force_mag * uz / u_mag
                            end
                        end
                    end
                end
                
                ux_eq = ux + 0.5f0 * Fx_wall * inv_rho
                uy_eq = uy + 0.5f0 * Fy_wall * inv_rho
                uz_eq = uz + 0.5f0 * Fz_wall * inv_rho
                usq_eq = ux_eq*ux_eq + uy_eq*uy_eq + uz_eq*uz_eq
                
                vel_out[x, y, z, b_idx, 1] = ux
                vel_out[x, y, z, b_idx, 2] = uy
                vel_out[x, y, z, b_idx, 3] = uz
                rho_out[x, y, z, b_idx] = rho

                
                
                
                g11, g12, g13, g21, g22, g23, g31, g32, g33 = compute_velocity_gradients(
                    vel_in, x, y, z, b_idx, block_size, neighbor_table
                )
                
                
                gsq11 = g11*g11 + g12*g21 + g13*g31
                gsq12 = g11*g12 + g12*g22 + g13*g32
                gsq13 = g11*g13 + g12*g23 + g13*g33
                gsq21 = g21*g11 + g22*g21 + g23*g31
                gsq22 = g21*g12 + g22*g22 + g23*g32
                gsq23 = g21*g13 + g22*g23 + g23*g33
                gsq31 = g31*g11 + g32*g21 + g33*g31
                gsq32 = g31*g12 + g32*g22 + g33*g32
                gsq33 = g31*g13 + g32*g23 + g33*g33
                
                tr_gsq = gsq11 + gsq22 + gsq33
                tr_term = tr_gsq / 3.0f0
                
                Sd11 = gsq11 - tr_term
                Sd22 = gsq22 - tr_term
                Sd33 = gsq33 - tr_term
                Sd12 = 0.5f0 * (gsq12 + gsq21)
                Sd13 = 0.5f0 * (gsq13 + gsq31)
                Sd23 = 0.5f0 * (gsq23 + gsq32)
                
                S12 = 0.5f0 * (g12 + g21)
                S13 = 0.5f0 * (g13 + g31)
                S23 = 0.5f0 * (g23 + g32)
                
                OP1 = Sd11*Sd11 + Sd22*Sd22 + Sd33*Sd33 + 2.0f0*(Sd12*Sd12 + Sd13*Sd13 + Sd23*Sd23)
                OP2 = g11*g11 + g22*g22 + g33*g33 + 2.0f0*(S12*S12 + S13*S13 + S23*S23)
                
                nu_eddy = 0.0f0
                if OP1 > 1.0f-12
                    OP1_32 = OP1 * sqrt(OP1)
                    OP2_52 = OP2 * OP2 * sqrt(max(OP2, 1.0f-12))
                    denom = OP2_52 + OP1 * sqrt(sqrt(max(OP1, 1.0f-12)))
                    if denom > 1.0f-12
                        nu_eddy = (c_wale * c_wale) * OP1_32 / denom
                    end
                end
                
                
                
                
                
                nu_eddy = max(nu_eddy, nu_sgs_background)
                
                tau_turb = tau_molecular + nu_eddy * 3.0f0
                omega = 1.0f0 / max(tau_turb, 0.500001f0)

                
                
                
                Pi_xx = 0.0f0; Pi_yy = 0.0f0; Pi_zz = 0.0f0
                Pi_xy = 0.0f0; Pi_yz = 0.0f0; Pi_zx = 0.0f0
                
                for k in Int32(1):Int32(27)
                    cx_f = Float32(cx_arr[k])
                    cy_f = Float32(cy_arr[k])
                    cz_f = Float32(cz_arr[k])
                    cu = cx_f*ux_eq + cy_f*uy_eq + cz_f*uz_eq
                    feq = rho * w_arr[k] * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - 1.5f0*usq_eq)
                    f_neq = f_stored[k] - feq
                    
                    Pi_xx += f_neq * cx_f * cx_f
                    Pi_yy += f_neq * cy_f * cy_f
                    Pi_zz += f_neq * cz_f * cz_f
                    Pi_xy += f_neq * cx_f * cy_f
                    Pi_yz += f_neq * cy_f * cz_f
                    Pi_zx += f_neq * cz_f * cx_f
                end

                for k in Int32(1):Int32(27)
                    cx_f = Float32(cx_arr[k])
                    cy_f = Float32(cy_arr[k])
                    cz_f = Float32(cz_arr[k])
                    w_k = w_arr[k]
                    
                    cu = cx_f*ux_eq + cy_f*uy_eq + cz_f*uz_eq
                    feq = rho * w_k * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - 1.5f0*usq_eq)
                    
                    force_term = w_k * 3.0f0 * (
                        (cx_f - ux + 3.0f0*cu*cx_f) * Fx_wall +
                        (cy_f - uy + 3.0f0*cu*cy_f) * Fy_wall +
                        (cz_f - uz + 3.0f0*cu*cz_f) * Fz_wall
                    )
                    
                    Q_xx = cx_f*cx_f - CS2_PHYSICS
                    Q_yy = cy_f*cy_f - CS2_PHYSICS
                    Q_zz = cz_f*cz_f - CS2_PHYSICS
                    
                    f_neq_reg = w_k * 4.5f0 * (
                        Pi_xx * Q_xx + Pi_yy * Q_yy + Pi_zz * Q_zz +
                        2.0f0 * (Pi_xy * cx_f*cy_f + Pi_yz * cy_f*cz_f + Pi_zx * cz_f*cx_f)
                    )
                    
                    f_coll = feq + (1.0f0 - omega) * f_neq_reg + (1.0f0 - 0.5f0*omega) * force_term
                    
                    if store_post_collision == Int32(1)
                        f_post_collision[x, y, z, b_idx, k] = f_coll
                    end
                    f_out[x, y, z, b_idx, k] = f_coll
                end
            end
        end
    end
end