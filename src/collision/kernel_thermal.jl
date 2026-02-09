# ==============================================================================
# THERMAL DDF KERNEL — Double Distribution Function for energy/temperature
# ==============================================================================
# Phase 2: Separate stream-collide kernel for the thermal (g) distributions.
# Uses simple BGK collision with relaxation rate omega_g = 1/tau_g.
# Reads rho and velocity from the flow kernel output (no recomputation).
#
# Thermal equilibrium: g_eq_k = w_k * T * (1 + 3*(c_k . u))
# Recovers: dT/dt + u·∇T = κ ∇²T   where κ = (tau_g - 0.5)/3
# ==============================================================================

using KernelAbstractions

@kernel function stream_collide_thermal!(
    g_out, g_in, g_post_collision,
    temperature_out, rho_arr, vel_arr,
    obstacle, sponge_arr,
    neighbor_table,
    active_coords_x, active_coords_y, active_coords_z,
    parent_g, parent_temperature, parent_vel, parent_ptr,
    parent_g_old, parent_temperature_old, parent_vel_old,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    tau_g::Float32,
    tau_g_parent::Float32,
    is_level_1::Int32,
    is_symmetric::Int32,
    nx_global::Int32, ny_global::Int32, nz_global::Int32,
    u_inlet::Float32,
    T_inlet::Float32,
    T_wall::Float32,
    wall_bc_adiabatic::Int32,   # 1=adiabatic (bounce-back), 0=isothermal (anti-bounce-back)
    n_blocks::Int32,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr, opp_arr,
    mirror_y_arr, mirror_z_arr,
    store_post_collision::Int32,
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

            # Read velocity from flow kernel output (already computed)
            ux = vel_arr[x, y, z, b_idx, 1]
            uy = vel_arr[x, y, z, b_idx, 2]
            uz = vel_arr[x, y, z, b_idx, 3]

            # ══════════════════════════════════════════════════════════
            # STEP 1: Pull-streaming — gather 27 g distributions
            # ══════════════════════════════════════════════════════════
            T_sum = 0.0f0

            for k in Int32(1):Int32(27)
                cx = cx_arr[k]
                cy = cy_arr[k]
                cz = cz_arr[k]

                sx = Int32(x) - cx
                sy = Int32(y) - cy
                sz = Int32(z) - cz

                val = 0.0f0

                if sx >= Int32(1) && sx <= block_size && sy >= Int32(1) && sy <= block_size && sz >= Int32(1) && sz <= block_size
                    val = g_in[sx, sy, sz, b_idx, k]
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
                        val = g_in[nsx, nsy, nsz, nb_idx_global, k]
                    else
                        src_gx = gx - cx
                        src_gy = gy - cy
                        src_gz = gz - cz

                        is_inlet  = src_gx < Int32(1)
                        is_outlet = src_gx > nx_global
                        is_y_min  = src_gy < Int32(1)
                        is_y_max  = src_gy > ny_global
                        is_z_min  = src_gz < Int32(1)
                        is_z_max  = src_gz > nz_global

                        if is_inlet
                            # Thermal equilibrium at inlet temperature
                            val = calculate_thermal_equilibrium(T_inlet, u_inlet, 0.0f0, 0.0f0,
                                      w_arr[k], Float32(cx), Float32(cy), Float32(cz))
                        elseif is_outlet
                            val = calculate_thermal_equilibrium(T_inlet, u_inlet, 0.0f0, 0.0f0,
                                      w_arr[k], Float32(cx), Float32(cy), Float32(cz))
                        elseif is_y_min && is_symmetric == Int32(1)
                            val = g_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif is_y_min || is_y_max
                            val = g_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif is_z_min || is_z_max
                            val = g_in[x, y, z, b_idx, mirror_z_arr[k]]
                        elseif is_level_1 == Int32(0)
                            # Coarse-to-fine interpolation for thermal field
                            val = interpolate_thermal_with_rescaling(
                                parent_g, parent_temperature, parent_vel,
                                parent_g_old, parent_temperature_old, parent_vel_old,
                                parent_ptr,
                                parent_dim_x, parent_dim_y, parent_dim_z,
                                src_gx, src_gy, src_gz,
                                k, block_size, w_arr[k],
                                Float32(cx), Float32(cy), Float32(cz),
                                tau_g_parent, tau_g,
                                temporal_weight,
                                use_temporal_interp
                            )
                        else
                            val = w_arr[k] * T_inlet
                        end
                    end
                end

                T_sum += val

                # ── Obstacle handling ──
                if is_obs
                    if wall_bc_adiabatic == Int32(1)
                        # Adiabatic: bounce-back (reflects g, preserves temperature)
                        g_out[x, y, z, b_idx, k] = g_in[x, y, z, b_idx, opp_arr[k]]
                    else
                        # Isothermal: anti-bounce-back to impose T_wall
                        g_eq_wall = calculate_thermal_equilibrium(T_wall, 0.0f0, 0.0f0, 0.0f0,
                                        w_arr[k], Float32(cx), Float32(cy), Float32(cz))
                        g_opp = g_in[x, y, z, b_idx, opp_arr[k]]
                        g_out[x, y, z, b_idx, k] = 2.0f0 * g_eq_wall - g_opp
                    end
                    if store_post_collision == Int32(1)
                        g_post_collision[x, y, z, b_idx, k] = g_out[x, y, z, b_idx, k]
                    end
                else
                    # Store streamed value temporarily (will be overwritten by collision below)
                    g_out[x, y, z, b_idx, k] = val
                end
            end

            # ══════════════════════════════════════════════════════════
            # STEP 2: Compute temperature + sponge + collision (non-obstacle)
            # ══════════════════════════════════════════════════════════
            if is_obs
                temperature_out[x, y, z, b_idx] = wall_bc_adiabatic == Int32(1) ? T_sum : T_wall
            else
                T_local = max(T_sum, 0.01f0)

                # Sponge blending for temperature
                sp = sponge_arr[x, y, z, b_idx]
                if sp > 0.0f0
                    T_local = T_local * (1.0f0 - sp) + T_inlet * sp
                    ux = ux * (1.0f0 - sp) + u_inlet * sp
                    uy = uy * (1.0f0 - sp)
                    uz = uz * (1.0f0 - sp)
                end

                temperature_out[x, y, z, b_idx] = T_local

                # BGK collision for thermal distributions
                omega_g = 1.0f0 / max(tau_g, 0.500001f0)

                for k in Int32(1):Int32(27)
                    g_streamed = g_out[x, y, z, b_idx, k]

                    # Sponge distribution blending
                    if sp > 0.0f0 && sponge_blend_distributions == Int32(1)
                        geq_target = calculate_thermal_equilibrium(T_inlet, u_inlet, 0.0f0, 0.0f0,
                                         w_arr[k], Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]))
                        g_streamed = g_streamed * (1.0f0 - sp) + geq_target * sp
                    end

                    geq_k = calculate_thermal_equilibrium(T_local, ux, uy, uz,
                                w_arr[k], Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]))

                    g_coll = g_streamed - omega_g * (g_streamed - geq_k)

                    if store_post_collision == Int32(1)
                        g_post_collision[x, y, z, b_idx, k] = g_coll
                    end
                    g_out[x, y, z, b_idx, k] = g_coll
                end
            end
        end
    end
end
