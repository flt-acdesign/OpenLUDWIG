using KernelAbstractions
using StaticArrays

@inline function chimera_forward(f_m::Float32, f_0::Float32, f_p::Float32, u::Float32)
    m0 = f_m + f_0 + f_p
    m1 = f_p - f_m
    m2 = f_m + f_p

    kc0 = m0
    kc1 = m1 - u * m0
    kc2 = m2 - 2.0f0 * u * m1 + u * u * m0

    return (kc0, kc1, kc2)
end

@inline function chimera_backward(kc0::Float32, kc1::Float32, kc2::Float32, u::Float32)
    m0 = kc0
    m1 = kc1 + u * kc0
    m2 = kc2 + 2.0f0 * u * kc1 + u * u * kc0

    f_p = (m2 + m1) * 0.5f0
    f_m = (m2 - m1) * 0.5f0
    f_0 = m0 - m2

    return (f_m, f_0, f_p)
end


@kernel function stream_collide_cumulant!(
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
    sponge_blend_distributions::Int32,
    # ── Phase 2A: configurable relaxation rates ──
    omega_bulk_param::Float32,
    omega_3_param::Float32,
    omega_4_param::Float32,
    # ── Phase 2B: adaptive ω₄ ──
    adaptive_omega4::Int32,
    lambda_param::Float32,
    # ── Phase 2C: limiter type (0=none, 1=factored, 2=positivity) ──
    limiter_type::Int32,
    # ── Compressibility correction: 0=off (standard O(u²)), 1=on (O(u⁴)) ──
    compressible_corr::Int32
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

            # ══════════════════════════════════════════════════════════
            # STEP 1: Pull-streaming — gather 27 distributions
            # ══════════════════════════════════════════════════════════
            rho = 0.0f0
            jx  = 0.0f0
            jy  = 0.0f0
            jz  = 0.0f0
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

                        is_inlet  = src_gx < Int32(1)
                        is_outlet = src_gx > nx_global
                        is_y_min  = src_gy < Int32(1)
                        is_y_max  = src_gy > ny_global
                        is_z_min  = src_gz < Int32(1)
                        is_z_max  = src_gz > nz_global

                        if is_inlet
                            noise = inlet_turbulence > 0.0f0 ? gradient_noise(gy, gz, time_step_seed, Int32(1234)) * inlet_turbulence * u_inlet : 0.0f0
                            u_inst = u_inlet + noise
                            val = calculate_equilibrium_auto(1.0f0, u_inst, 0.0f0, 0.0f0,
                                      w_arr[k], Float32(cx), Float32(cy), Float32(cz), compressible_corr)
                        elseif is_outlet
                            val = calculate_equilibrium_auto(1.0f0, u_inlet, 0.0f0, 0.0f0,
                                      w_arr[k], Float32(cx), Float32(cy), Float32(cz), compressible_corr)
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
                                use_temporal_interp,
                                compressible_corr
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

            # ══════════════════════════════════════════════════════════
            # Obstacle: simple bounce-back
            # ══════════════════════════════════════════════════════════
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
                # ══════════════════════════════════════════════════════
                # STEP 2: Macroscopic quantities + sponge + wall model
                # ══════════════════════════════════════════════════════
                rho = max(rho, 0.01f0)
                inv_rho = 1.0f0 / rho
                ux = jx * inv_rho
                uy = jy * inv_rho
                uz = jz * inv_rho

                sp = sponge_arr[x, y, z, b_idx]
                if sp > 0.0f0
                    rho_target = 1.0f0
                    ux_target  = u_inlet

                    rho = rho * (1.0f0 - sp) + rho_target * sp
                    ux  = ux  * (1.0f0 - sp) + ux_target  * sp
                    uy  = uy  * (1.0f0 - sp)
                    uz  = uz  * (1.0f0 - sp)

                    if sponge_blend_distributions == Int32(1)
                        for k in Int32(1):Int32(27)
                            feq_target = calculate_equilibrium_auto(rho_target, ux_target, 0.0f0, 0.0f0,
                                                               w_arr[k], Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]),
                                                               compressible_corr)
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
                        u_mag  = sqrt(ux*ux + uy*uy + uz*uz)
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
                            tau_res  = rho * nu_visc * (u_mag / dist_wall)

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

                vel_out[x, y, z, b_idx, 1] = ux
                vel_out[x, y, z, b_idx, 2] = uy
                vel_out[x, y, z, b_idx, 3] = uz
                rho_out[x, y, z, b_idx] = rho

                # ══════════════════════════════════════════════════════
                # STEP 3: WALE SGS → effective ω₁
                # ══════════════════════════════════════════════════════
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

                tr_gsq  = gsq11 + gsq22 + gsq33
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

                nu_mol  = (tau_molecular - 0.5f0) / 3.0f0
                nu_eff  = nu_mol + nu_eddy
                tau_eff = 0.5f0 + 3.0f0 * nu_eff
                omega1  = 1.0f0 / max(tau_eff, 0.500001f0)

                # ── Phase 2A: use configurable relaxation rates ──
                omega_bulk = omega_bulk_param
                omega3     = omega_3_param
                omega4     = omega_4_param

                # ── Phase 2B: Geier 2017 adaptive ω₄ for diagonal group ──
                omega4_diag = omega4     # default: same as base omega4
                if adaptive_omega4 == Int32(1)
                    # A = -(ω₁ - 2)² / (Λ · ω₁ + ε)
                    om1_m2 = omega1 - 2.0f0
                    denom_adaptive = lambda_param * omega1 + 1.0f-10
                    A_val = -(om1_m2 * om1_m2) / denom_adaptive
                    omega4_diag = clamp(A_val, 0.01f0, 2.0f0)
                end

                # ══════════════════════════════════════════════════════
                # STEP 4: Chimera forward transform (f → central moments)
                # ══════════════════════════════════════════════════════
                A = MVector{27, Float32}(undef)
                for k in Int32(1):Int32(27)
                    A[k] = f_stored[k]
                end

                # x-pass
                for iz in Int32(1):Int32(3)
                    for iy in Int32(1):Int32(3)
                        base = Int32(1) + (iy - Int32(1))*Int32(3) + (iz - Int32(1))*Int32(9)
                        fm = A[base]
                        f0 = A[base + Int32(1)]
                        fp = A[base + Int32(2)]
                        kc0, kc1, kc2 = chimera_forward(fm, f0, fp, ux_eq)
                        A[base]              = kc0
                        A[base + Int32(1)]   = kc1
                        A[base + Int32(2)]   = kc2
                    end
                end

                # y-pass
                for iz in Int32(1):Int32(3)
                    for ma in Int32(1):Int32(3)
                        i1 = ma + Int32(0)*Int32(3) + (iz - Int32(1))*Int32(9)
                        i2 = ma + Int32(1)*Int32(3) + (iz - Int32(1))*Int32(9)
                        i3 = ma + Int32(2)*Int32(3) + (iz - Int32(1))*Int32(9)
                        fm = A[i1]; f0 = A[i2]; fp = A[i3]
                        kc0, kc1, kc2 = chimera_forward(fm, f0, fp, uy_eq)
                        A[i1] = kc0; A[i2] = kc1; A[i3] = kc2
                    end
                end

                # z-pass
                for mb in Int32(0):Int32(2)
                    for ma in Int32(1):Int32(3)
                        i1 = ma + mb*Int32(3) + Int32(0)*Int32(9)
                        i2 = ma + mb*Int32(3) + Int32(1)*Int32(9)
                        i3 = ma + mb*Int32(3) + Int32(2)*Int32(9)
                        fm = A[i1]; f0 = A[i2]; fp = A[i3]
                        kc0, kc1, kc2 = chimera_forward(fm, f0, fp, uz_eq)
                        A[i1] = kc0; A[i2] = kc1; A[i3] = kc2
                    end
                end

                # ══════════════════════════════════════════════════════
                # STEP 5: Extract central moments → cumulants
                # ══════════════════════════════════════════════════════
                k_000 = A[1]

                k_200 = A[3];   k_020 = A[7];   k_002 = A[19]
                k_110 = A[5];   k_101 = A[11];  k_011 = A[13]

                k_210 = A[6];   k_120 = A[8];   k_201 = A[12]
                k_021 = A[16];  k_102 = A[20];  k_012 = A[22]
                k_111 = A[14]

                k_220 = A[9];   k_202 = A[21];  k_022 = A[25]
                k_211 = A[15];  k_121 = A[17];  k_112 = A[23]

                k_221 = A[18];  k_212 = A[24];  k_122 = A[26]

                k_222 = A[27]

                inv_r = 1.0f0 / max(k_000, 0.01f0)

                c200 = k_200 * inv_r;  c020 = k_020 * inv_r;  c002 = k_002 * inv_r
                c110 = k_110 * inv_r;  c101 = k_101 * inv_r;  c011 = k_011 * inv_r

                c210 = k_210 * inv_r;  c120 = k_120 * inv_r;  c201 = k_201 * inv_r
                c021 = k_021 * inv_r;  c102 = k_102 * inv_r;  c012 = k_012 * inv_r
                c111 = k_111 * inv_r

                c220n = k_220 * inv_r;  c202n = k_202 * inv_r;  c022n = k_022 * inv_r
                c211n = k_211 * inv_r;  c121n = k_121 * inv_r;  c112n = k_112 * inv_r

                c221n = k_221 * inv_r;  c212n = k_212 * inv_r;  c122n = k_122 * inv_r

                c222n = k_222 * inv_r

                # 4th-order cumulants (subtract products of 2nd-order)
                C_220 = c220n - c200*c020 - 2.0f0*c110*c110
                C_202 = c202n - c200*c002 - 2.0f0*c101*c101
                C_022 = c022n - c020*c002 - 2.0f0*c011*c011
                C_211 = c211n - c200*c011 - 2.0f0*c110*c101
                C_121 = c121n - c020*c101 - 2.0f0*c110*c011
                C_112 = c112n - c002*c110 - 2.0f0*c101*c011

                # ══════════════════════════════════════════════════════
                # STEP 6: Relaxation
                # ══════════════════════════════════════════════════════

                cs2 = 0.333333333f0

                # --- 2nd-order: trace (bulk) + deviatoric (ω₁) ---
                D_trace = c200 + c020 + c002
                D_trace_eq = 3.0f0 * cs2
                D_trace_s = (1.0f0 - omega_bulk) * D_trace + omega_bulk * D_trace_eq

                D1 = c200 - c020
                D2 = c200 - c002
                D1_s = (1.0f0 - omega1) * D1
                D2_s = (1.0f0 - omega1) * D2

                c200_s = (D_trace_s + D1_s + D2_s) / 3.0f0
                c020_s = (D_trace_s - 2.0f0*D1_s + D2_s) / 3.0f0
                c002_s = (D_trace_s + D1_s - 2.0f0*D2_s) / 3.0f0

                c110_s = (1.0f0 - omega1) * c110
                c101_s = (1.0f0 - omega1) * c101
                c011_s = (1.0f0 - omega1) * c011

                # ── Phase 2C-factored: Cauchy-Schwarz realizability limiter ──
                if limiter_type == Int32(1)
                    # Clamp off-diagonal 2nd-order cumulants to satisfy
                    # |c_αβ| ≤ √(c_αα · c_ββ)  (realizability)
                    bound_110 = sqrt(max(c200_s * c020_s, 0.0f0))
                    bound_101 = sqrt(max(c200_s * c002_s, 0.0f0))
                    bound_011 = sqrt(max(c020_s * c002_s, 0.0f0))
                    c110_s = clamp(c110_s, -bound_110, bound_110)
                    c101_s = clamp(c101_s, -bound_101, bound_101)
                    c011_s = clamp(c011_s, -bound_011, bound_011)
                end

                # --- 3rd-order: relax with ω₃ ---
                c210_s = (1.0f0 - omega3) * c210
                c120_s = (1.0f0 - omega3) * c120
                c201_s = (1.0f0 - omega3) * c201
                c021_s = (1.0f0 - omega3) * c021
                c102_s = (1.0f0 - omega3) * c102
                c012_s = (1.0f0 - omega3) * c012
                c111_s = (1.0f0 - omega3) * c111

                # --- 4th-order: relax cumulants ---
                # Phase 2B: diagonal group uses omega4_diag (adaptive or fixed)
                #           mixed   group uses omega4 (base value)
                C_220_s = (1.0f0 - omega4_diag) * C_220
                C_202_s = (1.0f0 - omega4_diag) * C_202
                C_022_s = (1.0f0 - omega4_diag) * C_022
                C_211_s = (1.0f0 - omega4) * C_211
                C_121_s = (1.0f0 - omega4) * C_121
                C_112_s = (1.0f0 - omega4) * C_112

                # ══════════════════════════════════════════════════════
                # STEP 7: Reconstruct central moments from cumulants
                # ══════════════════════════════════════════════════════
                c220_s = C_220_s + c200_s*c020_s + 2.0f0*c110_s*c110_s
                c202_s = C_202_s + c200_s*c002_s + 2.0f0*c101_s*c101_s
                c022_s = C_022_s + c020_s*c002_s + 2.0f0*c011_s*c011_s
                c211_s = C_211_s + c200_s*c011_s + 2.0f0*c110_s*c101_s
                c121_s = C_121_s + c020_s*c101_s + 2.0f0*c110_s*c011_s
                c112_s = C_112_s + c002_s*c110_s + 2.0f0*c101_s*c011_s

                c221_s = c200_s*c021_s + c020_s*c201_s +
                         4.0f0*c110_s*c111_s + 2.0f0*c101_s*c120_s + 2.0f0*c011_s*c210_s

                c212_s = c200_s*c012_s + c002_s*c210_s +
                         2.0f0*c110_s*c102_s + 4.0f0*c101_s*c111_s + 2.0f0*c011_s*c201_s

                c122_s = c020_s*c102_s + c002_s*c120_s +
                         2.0f0*c110_s*c012_s + 4.0f0*c011_s*c111_s + 2.0f0*c101_s*c021_s

                c222_s = c200_s*C_022_s + c020_s*C_202_s + c002_s*C_220_s +
                         4.0f0*(c110_s*C_112_s + c101_s*C_121_s + c011_s*C_211_s)
                c222_s += 2.0f0*(c210_s*c012_s + c201_s*c021_s + c120_s*c102_s) +
                          4.0f0*c111_s*c111_s
                c222_s += c200_s*c020_s*c002_s +
                          2.0f0*(c200_s*c011_s*c011_s + c020_s*c101_s*c101_s + c002_s*c110_s*c110_s) +
                          8.0f0*c110_s*c101_s*c011_s

                # ══════════════════════════════════════════════════════
                # STEP 8: Write post-collision central moments back
                # ══════════════════════════════════════════════════════
                r = k_000

                A[1] = r

                A[2]  = 0.0f0;  A[4]  = 0.0f0;  A[10] = 0.0f0

                A[3]  = r * c200_s;   A[7]  = r * c020_s;   A[19] = r * c002_s
                A[5]  = r * c110_s;   A[11] = r * c101_s;   A[13] = r * c011_s

                A[6]  = r * c210_s;   A[8]  = r * c120_s;   A[12] = r * c201_s
                A[16] = r * c021_s;   A[20] = r * c102_s;   A[22] = r * c012_s
                A[14] = r * c111_s

                A[9]  = r * c220_s;   A[21] = r * c202_s;   A[25] = r * c022_s
                A[15] = r * c211_s;   A[17] = r * c121_s;   A[23] = r * c112_s

                A[18] = r * c221_s;   A[24] = r * c212_s;   A[26] = r * c122_s

                A[27] = r * c222_s

                # ══════════════════════════════════════════════════════
                # STEP 9: Chimera backward transform (central moments → f)
                # ══════════════════════════════════════════════════════

                # z-pass backward
                for mb in Int32(0):Int32(2)
                    for ma in Int32(1):Int32(3)
                        i1 = ma + mb*Int32(3) + Int32(0)*Int32(9)
                        i2 = ma + mb*Int32(3) + Int32(1)*Int32(9)
                        i3 = ma + mb*Int32(3) + Int32(2)*Int32(9)
                        kc0 = A[i1]; kc1 = A[i2]; kc2 = A[i3]
                        fm, f0, fp = chimera_backward(kc0, kc1, kc2, uz_eq)
                        A[i1] = fm; A[i2] = f0; A[i3] = fp
                    end
                end

                # y-pass backward
                for iz in Int32(1):Int32(3)
                    for ma in Int32(1):Int32(3)
                        i1 = ma + Int32(0)*Int32(3) + (iz - Int32(1))*Int32(9)
                        i2 = ma + Int32(1)*Int32(3) + (iz - Int32(1))*Int32(9)
                        i3 = ma + Int32(2)*Int32(3) + (iz - Int32(1))*Int32(9)
                        kc0 = A[i1]; kc1 = A[i2]; kc2 = A[i3]
                        fm, f0, fp = chimera_backward(kc0, kc1, kc2, uy_eq)
                        A[i1] = fm; A[i2] = f0; A[i3] = fp
                    end
                end

                # x-pass backward
                for iz in Int32(1):Int32(3)
                    for iy in Int32(1):Int32(3)
                        base = Int32(1) + (iy - Int32(1))*Int32(3) + (iz - Int32(1))*Int32(9)
                        kc0 = A[base]; kc1 = A[base + Int32(1)]; kc2 = A[base + Int32(2)]
                        fm, f0, fp = chimera_backward(kc0, kc1, kc2, ux_eq)
                        A[base]            = fm
                        A[base + Int32(1)] = f0
                        A[base + Int32(2)] = fp
                    end
                end

                # ══════════════════════════════════════════════════════
                # Phase 2C-positivity: enforce non-negative distributions
                # ══════════════════════════════════════════════════════
                if limiter_type == Int32(2)
                    # Find minimum post-collision distribution
                    f_min_val = A[1]
                    for k in Int32(2):Int32(27)
                        f_min_val = min(f_min_val, A[k])
                    end

                    if f_min_val < 0.0f0
                        # Compute equilibrium and find safe scaling factor α
                        alpha = 1.0f0
                        for k in Int32(1):Int32(27)
                            cx_f = Float32(cx_arr[k])
                            cy_f = Float32(cy_arr[k])
                            cz_f = Float32(cz_arr[k])
                            feq_k = calculate_equilibrium_auto(rho, ux_eq, uy_eq, uz_eq,
                                        w_arr[k], cx_f, cy_f, cz_f, compressible_corr)
                            fneq_k = A[k] - feq_k
                            # If non-equilibrium drives f negative, limit it
                            if fneq_k < 0.0f0 && feq_k > 1.0f-15
                                alpha_k = feq_k / (feq_k - fneq_k + 1.0f-15)
                                alpha = min(alpha, alpha_k)
                            end
                        end
                        alpha = max(alpha, 0.0f0)  # safety clamp

                        # Re-apply: f = feq + alpha * fneq
                        for k in Int32(1):Int32(27)
                            cx_f = Float32(cx_arr[k])
                            cy_f = Float32(cy_arr[k])
                            cz_f = Float32(cz_arr[k])
                            feq_k = calculate_equilibrium_auto(rho, ux_eq, uy_eq, uz_eq,
                                        w_arr[k], cx_f, cy_f, cz_f, compressible_corr)
                            A[k] = feq_k + alpha * (A[k] - feq_k)
                        end
                    end
                end

                # ══════════════════════════════════════════════════════
                # STEP 10: Guo forcing + write output
                # ══════════════════════════════════════════════════════
                for k in Int32(1):Int32(27)
                    cx_f = Float32(cx_arr[k])
                    cy_f = Float32(cy_arr[k])
                    cz_f = Float32(cz_arr[k])
                    w_k  = w_arr[k]

                    cu = cx_f*ux_eq + cy_f*uy_eq + cz_f*uz_eq
                    force_term = w_k * 3.0f0 * (
                        (cx_f - ux + 3.0f0*cu*cx_f) * Fx_wall +
                        (cy_f - uy + 3.0f0*cu*cy_f) * Fy_wall +
                        (cz_f - uz + 3.0f0*cu*cz_f) * Fz_wall
                    )

                    f_coll = A[k] + (1.0f0 - 0.5f0*omega1) * force_term

                    if store_post_collision == Int32(1)
                        f_post_collision[x, y, z, b_idx, k] = f_coll
                    end
                    f_out[x, y, z, b_idx, k] = f_coll
                end
            end
        end
    end
end