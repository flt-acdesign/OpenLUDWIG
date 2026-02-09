using Printf
using KernelAbstractions
using Atomix


mutable struct ForceAccumulator
    sum_Fx_p::Float64;  sum_Fy_p::Float64;  sum_Fz_p::Float64
    sum_Fx_v::Float64;  sum_Fy_v::Float64;  sum_Fz_v::Float64
    sum_Mx::Float64;    sum_My::Float64;    sum_Mz::Float64

    min_Cd::Float64;    max_Cd::Float64
    min_Cl::Float64;    max_Cl::Float64
    min_Cs::Float64;    max_Cs::Float64

    rho_ref::Float64
    u_ref::Float64
    area_ref::Float64
    chord_ref::Float64

    n_samples::Int
end

function ForceAccumulator(fd)
    ForceAccumulator(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        Inf, -Inf,
        Inf, -Inf,
        Inf, -Inf,
        fd.rho_ref, fd.u_ref, fd.area_ref, fd.chord_ref,
        0
    )
end

function accumulate_forces!(acc::ForceAccumulator, fd)
    acc.sum_Fx_p += fd.Fx_pressure
    acc.sum_Fy_p += fd.Fy_pressure
    acc.sum_Fz_p += fd.Fz_pressure
    acc.sum_Fx_v += fd.Fx_viscous
    acc.sum_Fy_v += fd.Fy_viscous
    acc.sum_Fz_v += fd.Fz_viscous
    acc.sum_Mx   += fd.Mx
    acc.sum_My   += fd.My
    acc.sum_Mz   += fd.Mz

    acc.min_Cd = min(acc.min_Cd, fd.Cd)
    acc.max_Cd = max(acc.max_Cd, fd.Cd)
    acc.min_Cl = min(acc.min_Cl, fd.Cl)
    acc.max_Cl = max(acc.max_Cl, fd.Cl)
    acc.min_Cs = min(acc.min_Cs, fd.Cs)
    acc.max_Cs = max(acc.max_Cs, fd.Cs)

    acc.n_samples += 1
    return nothing
end

function print_averaged_summary(acc::ForceAccumulator; re_number::Float64=0.0,
                                 ma_physical::Float64=0.0, ma_lattice::Float64=0.0,
                                 compressibility_correction::Bool=false)
    n = acc.n_samples
    if n == 0
        println("[Forces] No samples accumulated (all steps during ramp?)")
        return
    end

    inv_n = 1.0 / n

    avg_Fx_p = acc.sum_Fx_p * inv_n
    avg_Fy_p = acc.sum_Fy_p * inv_n
    avg_Fz_p = acc.sum_Fz_p * inv_n
    avg_Fx_v = acc.sum_Fx_v * inv_n
    avg_Fy_v = acc.sum_Fy_v * inv_n
    avg_Fz_v = acc.sum_Fz_v * inv_n
    avg_Mx   = acc.sum_Mx * inv_n
    avg_My   = acc.sum_My * inv_n
    avg_Mz   = acc.sum_Mz * inv_n

    avg_Fx = avg_Fx_p + avg_Fx_v
    avg_Fy = avg_Fy_p + avg_Fy_v
    avg_Fz = avg_Fz_p + avg_Fz_v

    q_inf = 0.5 * acc.rho_ref * acc.u_ref^2
    F_ref = q_inf * acc.area_ref
    M_ref = F_ref * acc.chord_ref

    avg_Cd  = F_ref > 1e-10 ? avg_Fx / F_ref : 0.0
    avg_Cl  = F_ref > 1e-10 ? avg_Fz / F_ref : 0.0
    avg_Cs  = F_ref > 1e-10 ? avg_Fy / F_ref : 0.0
    avg_Cdp = F_ref > 1e-10 ? avg_Fx_p / F_ref : 0.0
    avg_Cdv = F_ref > 1e-10 ? avg_Fx_v / F_ref : 0.0
    avg_Cmx = M_ref > 1e-10 ? avg_Mx / M_ref : 0.0
    avg_Cmy = M_ref > 1e-10 ? avg_My / M_ref : 0.0
    avg_Cmz = M_ref > 1e-10 ? avg_Mz / M_ref : 0.0

    pct_pressure = abs(avg_Fx) > 1e-10 ? 100.0 * abs(avg_Fx_p) / (abs(avg_Fx_p) + abs(avg_Fx_v)) : 0.0

    println()
    println("============================================================")
    println("          TIME-AVERAGED AERODYNAMIC FORCES")
    @printf("          (%d samples after ramp)\n", n)
    if compressibility_correction
        println("          (Compressible solver: O(u⁴) equilibrium + DDF)")
    end
    println("============================================================")
    println()
    println("Reference Values:")
    @printf("  Re     = %.2e\n", re_number)
    @printf("  Ma     = %.4f (physical), %.4f (lattice)\n", ma_physical, ma_lattice)
    @printf("  ρ_ref  = %.4f kg/m³\n", acc.rho_ref)
    @printf("  U_ref  = %.4f m/s\n", acc.u_ref)
    @printf("  A_ref  = %.4f m²\n", acc.area_ref)
    @printf("  L_ref  = %.4f m\n", acc.chord_ref)
    @printf("  q_∞    = %.4f Pa\n", q_inf)
    println()
    println("Averaged Forces [N]:")
    @printf("  Fx (drag)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n", avg_Fx, avg_Fx_p, avg_Fx_v)
    @printf("  Fy (side)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n", avg_Fy, avg_Fy_p, avg_Fy_v)
    @printf("  Fz (lift)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n", avg_Fz, avg_Fz_p, avg_Fz_v)
    println()
    println("Averaged Moments [N·m]:")
    @printf("  Mx (roll)  = %+.4e\n", avg_Mx)
    @printf("  My (pitch) = %+.4e\n", avg_My)
    @printf("  Mz (yaw)   = %+.4e\n", avg_Mz)
    println()
    println("Averaged Coefficients:")
    @printf("  Cd  = %+.6f  (Cdp = %+.6f, Cdv = %+.6f)\n", avg_Cd, avg_Cdp, avg_Cdv)
    @printf("  Cl  = %+.6f\n", avg_Cl)
    @printf("  Cs  = %+.6f\n", avg_Cs)
    @printf("  Cmy = %+.6f\n", avg_Cmy)
    println()
    println("Coefficient Ranges (instantaneous min / max):")
    @printf("  Cd  ∈ [%+.6f, %+.6f]\n", acc.min_Cd, acc.max_Cd)
    @printf("  Cl  ∈ [%+.6f, %+.6f]\n", acc.min_Cl, acc.max_Cl)
    @printf("  Cs  ∈ [%+.6f, %+.6f]\n", acc.min_Cs, acc.max_Cs)
    println()
    @printf("Drag breakdown: %.1f%% pressure, %.1f%% viscous\n", pct_pressure, 100.0 - pct_pressure)
    println("============================================================")
end



@inline function compute_stress_from_cell(
    rho_val::Float32,
    ux::Float32, uy::Float32, uz::Float32,
    nx::Float32, ny::Float32, nz::Float32,
    dist_to_wall::Float32,
    tau_molecular::Float32,
    pressure_scale::Float32,
    stress_scale::Float32,
    c_wale::Float32=0.0f0,
    nu_sgs_bg::Float32=0.0f0
)
    p_gauge_lat = (rho_val - 1.0f0) / 3.0f0
    p_phys = p_gauge_lat * pressure_scale

    u_dot_n = ux * nx + uy * ny + uz * nz
    ut_x = ux - u_dot_n * nx
    ut_y = uy - u_dot_n * ny
    ut_z = uz - u_dot_n * nz

    u_tan_mag = sqrt(ut_x * ut_x + ut_y * ut_y + ut_z * ut_z)

    nu_mol = (tau_molecular - 0.5f0) / 3.0f0

    # Estimate SGS viscosity from local shear rate: |S| ≈ u_tan / d_wall
    # WALE model: nu_sgs = (Cw·Δ)² · |S|  (simplified; Δ = 1 in lattice units)
    nu_sgs = 0.0f0
    if c_wale > 0.0f0 && dist_to_wall > 0.01f0 && u_tan_mag > 1.0f-10
        S_mag = u_tan_mag / dist_to_wall
        nu_sgs = (c_wale * c_wale) * S_mag
    end
    nu_sgs = max(nu_sgs, nu_sgs_bg)
    nu_eff = nu_mol + nu_sgs

    tau_x = 0.0f0
    tau_y = 0.0f0
    tau_z = 0.0f0

    if u_tan_mag > 1.0f-10 && dist_to_wall > 0.01f0
        tau_lat_mag = rho_val * nu_eff * u_tan_mag / dist_to_wall
        tau_phys_mag = tau_lat_mag * stress_scale

        tau_x = (ut_x / u_tan_mag) * tau_phys_mag
        tau_y = (ut_y / u_tan_mag) * tau_phys_mag
        tau_z = (ut_z / u_tan_mag) * tau_phys_mag
    end

    return (p_phys, tau_x, tau_y, tau_z)
end

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


@inline function trilinear_sample_flow(
    rho_arr, vel_arr, obstacle_arr, block_ptr,
    px::Float32, py::Float32, pz::Float32,
    dx::Float32, block_size::Int32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32
)
    inv_dx = 1.0f0 / dx
    fx = px * inv_dx - 0.5f0
    fy = py * inv_dx - 0.5f0
    fz = pz * inv_dx - 0.5f0

    ix0 = floor(Int32, fx) + Int32(1)
    iy0 = floor(Int32, fy) + Int32(1)
    iz0 = floor(Int32, fz) + Int32(1)

    wx = clamp(fx - Float32(ix0 - Int32(1)), 0.0f0, 1.0f0)
    wy = clamp(fy - Float32(iy0 - Int32(1)), 0.0f0, 1.0f0)
    wz = clamp(fz - Float32(iz0 - Int32(1)), 0.0f0, 1.0f0)

    w0x = 1.0f0 - wx;  w1x = wx
    w0y = 1.0f0 - wy;  w1y = wy
    w0z = 1.0f0 - wz;  w1z = wz

    rho_acc = 0.0f0
    ux_acc  = 0.0f0
    uy_acc  = 0.0f0
    uz_acc  = 0.0f0
    w_acc   = 0.0f0

    gx = ix0;              gy = iy0;              gz = iz0
    w_k = w0x * w0y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0;              gz = iz0
    w_k = w1x * w0y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0;              gy = iy0 + Int32(1);  gz = iz0
    w_k = w0x * w1y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0 + Int32(1);  gz = iz0
    w_k = w1x * w1y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0;              gy = iy0;              gz = iz0 + Int32(1)
    w_k = w0x * w0y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0;              gz = iz0 + Int32(1)
    w_k = w1x * w0y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0;              gy = iy0 + Int32(1);  gz = iz0 + Int32(1)
    w_k = w0x * w1y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0 + Int32(1);  gz = iz0 + Int32(1)
    w_k = w1x * w1y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    if w_acc > 0.01f0
        inv_w = 1.0f0 / w_acc
        return (rho_acc * inv_w, ux_acc * inv_w, uy_acc * inv_w, uz_acc * inv_w, true, w_acc)
    end
    return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false, 0.0f0)
end



@kernel function map_stresses_kernel!(
    p_map, sx_map, sy_map, sz_map,
    cx_tri, cy_tri, cz_tri,
    nx_tri, ny_tri, nz_tri,
    rho, vel, obstacle, block_ptr,
    block_size::Int32, dx::Float32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32,
    mesh_offset_x::Float32, mesh_offset_y::Float32, mesh_offset_z::Float32,
    pressure_scale::Float32, stress_scale::Float32,
    tau_molecular::Float32,
    search_radius::Int32,
    c_wale::Float32,
    nu_sgs_bg::Float32
)
    i = @index(Global)

    @inbounds begin
        tx = cx_tri[i] + mesh_offset_x
        ty = cy_tri[i] + mesh_offset_y
        tz = cz_tri[i] + mesh_offset_z

        n_x = nx_tri[i]
        n_y = ny_tri[i]
        n_z = nz_tri[i]

        best_rho       = 1.0f0
        best_ux        = 0.0f0
        best_uy        = 0.0f0
        best_uz        = 0.0f0
        best_wall_dist = 1.0f0
        best_quality   = 0.0f0
        found_fluid    = false

        # Multi-sample probe: try all distances, keep closest with good quality
        # "Good" = w_acc >= 0.5 (at least half the trilinear stencil in fluid)
        min_quality = 0.5f0

        for probe_i in Int32(1):search_radius
            d_cells = Float32(probe_i) + 0.5f0

            probe_x = tx + n_x * d_cells * dx
            probe_y = ty + n_y * d_cells * dx
            probe_z = tz + n_z * d_cells * dx

            r, ux, uy, uz, ok, w_q = trilinear_sample_flow(
                rho, vel, obstacle, block_ptr,
                probe_x, probe_y, probe_z,
                dx, block_size, dim_x, dim_y, dim_z
            )

            if ok && w_q >= min_quality
                # Good quality sample — take it (closest good probe wins)
                best_rho       = r
                best_ux        = ux
                best_uy        = uy
                best_uz        = uz
                best_wall_dist = d_cells
                best_quality   = w_q
                found_fluid    = true
                break
            elseif ok && w_q > best_quality
                # Below threshold but better than what we have — keep as candidate
                best_rho       = r
                best_ux        = ux
                best_uy        = uy
                best_uz        = uz
                best_wall_dist = d_cells
                best_quality   = w_q
                found_fluid    = true
                # Don't break — keep searching for a good quality probe
            end
        end

        if !found_fluid
            gx_f = tx / dx
            gy_f = ty / dx
            gz_f = tz / dx
            g_x = floor(Int32, gx_f) + Int32(1)
            g_y = floor(Int32, gy_f) + Int32(1)
            g_z = floor(Int32, gz_f) + Int32(1)

            best_dist_sq = Float32(1e10)

            for radius in Int32(1):search_radius
                if found_fluid
                    break
                end
                for dz in -radius:radius
                    for dy in -radius:radius
                        for ddx in -radius:radius
                            at_shell = (abs(ddx) == radius) || (abs(dy) == radius) || (abs(dz) == radius)
                            if !at_shell
                                continue
                            end

                            check_gx = g_x + Int32(ddx)
                            check_gy = g_y + Int32(dy)
                            check_gz = g_z + Int32(dz)

                            b_idx, lx, ly, lz, valid = get_cell_at_position(
                                check_gx, check_gy, check_gz,
                                block_ptr, block_size, dim_x, dim_y, dim_z
                            )

                            if valid && !obstacle[lx, ly, lz, b_idx]
                                cell_cx = (Float32(check_gx) - 0.5f0) * dx
                                cell_cy = (Float32(check_gy) - 0.5f0) * dx
                                cell_cz = (Float32(check_gz) - 0.5f0) * dx

                                disp_x = cell_cx - tx
                                disp_y = cell_cy - ty
                                disp_z = cell_cz - tz

                                dot_n = disp_x * n_x + disp_y * n_y + disp_z * n_z
                                if dot_n <= 0.0f0
                                    continue
                                end

                                dist_sq = disp_x^2 + disp_y^2 + disp_z^2
                                if dist_sq < best_dist_sq
                                    best_dist_sq = dist_sq
                                    best_rho = rho[lx, ly, lz, b_idx]
                                    best_ux  = vel[lx, ly, lz, b_idx, 1]
                                    best_uy  = vel[lx, ly, lz, b_idx, 2]
                                    best_uz  = vel[lx, ly, lz, b_idx, 3]
                                    best_wall_dist = dot_n / dx
                                    found_fluid = true
                                end
                            end
                        end
                    end
                end
            end
        end

        p_val = 0.0f0
        tau_x = 0.0f0
        tau_y = 0.0f0
        tau_z = 0.0f0

        if found_fluid
            wall_dist = max(best_wall_dist, 0.5f0)

            p_val, tau_x, tau_y, tau_z = compute_stress_from_cell(
                best_rho, best_ux, best_uy, best_uz,
                n_x, n_y, n_z,
                wall_dist, tau_molecular,
                pressure_scale, stress_scale,
                c_wale, nu_sgs_bg
            )
        end

        p_map[i] = p_val
        sx_map[i] = tau_x
        sy_map[i] = tau_y
        sz_map[i] = tau_z
    end
end



@kernel function map_minus_side_only_kernel!(
    p_map_minus, sx_map_minus, sy_map_minus, sz_map_minus,
    cx_tri, cy_tri, cz_tri,
    nx_tri, ny_tri, nz_tri,
    rho, vel, obstacle, block_ptr,
    block_size::Int32, dx::Float32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32,
    mesh_offset_x::Float32, mesh_offset_y::Float32, mesh_offset_z::Float32,
    pressure_scale::Float32, stress_scale::Float32,
    tau_molecular::Float32,
    search_radius::Int32,
    c_wale::Float32,
    nu_sgs_bg::Float32
)
    i = @index(Global)

    @inbounds begin
        tx = cx_tri[i] + mesh_offset_x
        ty = cy_tri[i] + mesh_offset_y
        tz = cz_tri[i] + mesh_offset_z

        n_x = nx_tri[i]
        n_y = ny_tri[i]
        n_z = nz_tri[i]

        # --- Probe only the MINUS side (opposite to outward normal) ---
        minus_rho       = 1.0f0
        minus_ux        = 0.0f0
        minus_uy        = 0.0f0
        minus_uz        = 0.0f0
        minus_wall_dist = 1.0f0
        minus_quality   = 0.0f0
        found_minus     = false

        min_quality = 0.5f0

        # Ray-march along -normal, starting at 1.5*dx (matches proven plus-side range)
        for probe_i in Int32(1):search_radius
            d_cells = Float32(probe_i) + 0.5f0

            probe_x = tx - n_x * d_cells * dx
            probe_y = ty - n_y * d_cells * dx
            probe_z = tz - n_z * d_cells * dx

            r, ux, uy, uz, ok, w_q = trilinear_sample_flow(
                rho, vel, obstacle, block_ptr,
                probe_x, probe_y, probe_z,
                dx, block_size, dim_x, dim_y, dim_z
            )

            if ok && w_q >= min_quality
                minus_rho       = r
                minus_ux        = ux
                minus_uy        = uy
                minus_uz        = uz
                minus_wall_dist = d_cells
                minus_quality   = w_q
                found_minus     = true
                break
            elseif ok && w_q > minus_quality
                minus_rho       = r
                minus_ux        = ux
                minus_uy        = uy
                minus_uz        = uz
                minus_wall_dist = d_cells
                minus_quality   = w_q
                found_minus     = true
            end
        end

        # Brute-force fallback: find nearest fluid cell on the minus side
        if !found_minus
            gx_f = tx / dx
            gy_f = ty / dx
            gz_f = tz / dx
            g_x = floor(Int32, gx_f) + Int32(1)
            g_y = floor(Int32, gy_f) + Int32(1)
            g_z = floor(Int32, gz_f) + Int32(1)

            best_dist_sq = Float32(1e10)

            for radius in Int32(1):search_radius
                if found_minus; break; end
                for dz in -radius:radius
                    for dy in -radius:radius
                        for ddx in -radius:radius
                            at_shell = (abs(ddx) == radius) || (abs(dy) == radius) || (abs(dz) == radius)
                            if !at_shell; continue; end

                            check_gx = g_x + Int32(ddx)
                            check_gy = g_y + Int32(dy)
                            check_gz = g_z + Int32(dz)

                            b_idx, lx, ly, lz, valid = get_cell_at_position(
                                check_gx, check_gy, check_gz,
                                block_ptr, block_size, dim_x, dim_y, dim_z
                            )

                            if valid && !obstacle[lx, ly, lz, b_idx]
                                cell_cx = (Float32(check_gx) - 0.5f0) * dx
                                cell_cy = (Float32(check_gy) - 0.5f0) * dx
                                cell_cz = (Float32(check_gz) - 0.5f0) * dx

                                disp_x = cell_cx - tx
                                disp_y = cell_cy - ty
                                disp_z = cell_cz - tz

                                # Only accept cells on the MINUS side (dot < 0)
                                dot_n = disp_x * n_x + disp_y * n_y + disp_z * n_z
                                if dot_n >= 0.0f0; continue; end

                                dist_sq = disp_x^2 + disp_y^2 + disp_z^2
                                if dist_sq < best_dist_sq
                                    best_dist_sq = dist_sq
                                    minus_rho = rho[lx, ly, lz, b_idx]
                                    minus_ux  = vel[lx, ly, lz, b_idx, 1]
                                    minus_uy  = vel[lx, ly, lz, b_idx, 2]
                                    minus_uz  = vel[lx, ly, lz, b_idx, 3]
                                    minus_wall_dist = abs(dot_n) / dx
                                    found_minus = true
                                end
                            end
                        end
                    end
                end
            end
        end

        # Compute stress from minus-side fluid (zero if no fluid found — correct for closed bodies)
        p_minus  = 0.0f0
        tx_minus = 0.0f0
        ty_minus = 0.0f0
        tz_minus = 0.0f0

        if found_minus
            wall_dist_m = max(minus_wall_dist, 0.5f0)
            p_minus, tx_minus, ty_minus, tz_minus = compute_stress_from_cell(
                minus_rho, minus_ux, minus_uy, minus_uz,
                -n_x, -n_y, -n_z,
                wall_dist_m, tau_molecular,
                pressure_scale, stress_scale,
                c_wale, nu_sgs_bg
            )
        end

        p_map_minus[i]  = p_minus
        sx_map_minus[i] = tx_minus
        sy_map_minus[i] = ty_minus
        sz_map_minus[i] = tz_minus
    end
end



@kernel function compute_net_triangle_forces_kernel!(
    net_Fx, net_Fy, net_Fz,
    p_map_plus, sx_map_plus, sy_map_plus, sz_map_plus,
    p_map_minus, sx_map_minus, sy_map_minus, sz_map_minus,
    nx_tri, ny_tri, nz_tri,
    areas
)
    i = @index(Global)

    @inbounds begin
        p_net = p_map_plus[i] - p_map_minus[i]

        sx_net = sx_map_plus[i] - sx_map_minus[i]
        sy_net = sy_map_plus[i] - sy_map_minus[i]
        sz_net = sz_map_plus[i] - sz_map_minus[i]

        nx = nx_tri[i]
        ny = ny_tri[i]
        nz = nz_tri[i]
        A  = areas[i]

        Fp_x = -p_net * nx * A
        Fp_y = -p_net * ny * A
        Fp_z = -p_net * nz * A

        Fv_x = sx_net * A
        Fv_y = sy_net * A
        Fv_z = sz_net * A

        net_Fx[i] = Fp_x + Fv_x
        net_Fy[i] = Fp_y + Fv_y
        net_Fz[i] = Fp_z + Fv_z
    end
end



@kernel function integrate_forces_kernel!(
    Fx_p_acc, Fy_p_acc, Fz_p_acc,
    Fx_v_acc, Fy_v_acc, Fz_v_acc,
    Mx_acc, My_acc, Mz_acc,
    p_map, sx_map, sy_map, sz_map,
    cx_tri, cy_tri, cz_tri,
    nx_tri, ny_tri, nz_tri,
    areas,
    offset_x::Float32, offset_y::Float32, offset_z::Float32,
    ref_x::Float32, ref_y::Float32, ref_z::Float32
)
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

        cx = cx_tri[i] + offset_x
        cy = cy_tri[i] + offset_y
        cz = cz_tri[i] + offset_z

        dFp_x = -p * nx * A
        dFp_y = -p * ny * A
        dFp_z = -p * nz * A

        dFv_x = tau_x * A
        dFv_y = tau_y * A
        dFv_z = tau_z * A

        dFx = dFp_x + dFv_x
        dFy = dFp_y + dFv_y
        dFz = dFp_z + dFv_z

        rx = cx - ref_x
        ry = cy - ref_y
        rz = cz - ref_z

        dMx = ry * dFz - rz * dFy
        dMy = rz * dFx - rx * dFz
        dMz = rx * dFy - ry * dFx

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



function map_surface_stresses!(force_data::ForceData, level, gpu_mesh,
                               backend, params; search_radius::Int=5)

    mesh_offset = params.mesh_offset
    tau_mol = level.tau

    velocity_scale = params.velocity_scale
    rho_phys = params.rho_physical

    pressure_scale = Float32(rho_phys * velocity_scale * velocity_scale)
    stress_scale = pressure_scale

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
        Float32(C_SMAGO),
        Float32(NU_SGS_BACKGROUND),
        ndrange=(gpu_mesh.n_triangles,)
    )

    KernelAbstractions.synchronize(backend)

    if !force_data.diagnostics_printed
        p_cpu = Array(force_data.pressure_map)
        sx_cpu = Array(force_data.shear_x_map)

        n_total = gpu_mesh.n_triangles
        n_nonzero_p = count(x -> abs(x) > 1e-10, p_cpu)
        n_nonzero_s = count(x -> abs(x) > 1e-10, sx_cpu)

        println("[Surface] Mapped $(n_total) triangles (quality-aware trilinear interpolation):")
        println("          Mesh offset: ($(round(mesh_offset[1], digits=3)), $(round(mesh_offset[2], digits=3)), $(round(mesh_offset[3], digits=3)))")
        println("          Grid spacing dx = $(round(level.dx, digits=6)) m")
        println("          Pressure scale = $(round(pressure_scale, digits=2)) Pa")
        println("          Coverage: pressure=$(n_nonzero_p)/$(n_total) ($(round(100*n_nonzero_p/n_total, digits=1))%)")

        if n_nonzero_p > 0
            p_nz = filter(x -> abs(x) > 1e-10, p_cpu)
            p_abs = abs.(p_nz)
            p_mean = sum(p_abs) / length(p_abs)
            p_std = sqrt(sum((p_abs .- p_mean).^2) / length(p_abs))
            @printf("          Pressure range: [%.3e, %.3e] Pa\n", minimum(p_nz), maximum(p_nz))
            @printf("          Pressure |p|: mean=%.3e, std=%.3e, CoV=%.1f%%\n",
                    p_mean, p_std, 100.0 * p_std / max(p_mean, 1e-20))
            @printf("          Pressure max/min |p| ratio: %.1f\n",
                    maximum(p_abs) / max(minimum(p_abs), 1e-20))
        end

        force_data.diagnostics_printed = true
    end
end

function integrate_surface_forces!(force_data::ForceData, gpu_mesh, backend, params)
    n_tri = gpu_mesh.n_triangles

    reset_forces!(force_data)

    Fx_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fy_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fz_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fx_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fy_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fz_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mx_acc = KernelAbstractions.zeros(backend, Float32, 1)
    My_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mz_acc = KernelAbstractions.zeros(backend, Float32, 1)

    mc = force_data.moment_center
    mesh_offset = params.mesh_offset

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

    Fx_p = Float64(Array(Fx_p_acc)[1])
    Fy_p = Float64(Array(Fy_p_acc)[1])
    Fz_p = Float64(Array(Fz_p_acc)[1])
    Fx_v = Float64(Array(Fx_v_acc)[1])
    Fy_v = Float64(Array(Fy_v_acc)[1])
    Fz_v = Float64(Array(Fz_v_acc)[1])
    Mx = Float64(Array(Mx_acc)[1])
    My = Float64(Array(My_acc)[1])
    Mz = Float64(Array(Mz_acc)[1])

    if force_data.symmetric
        Fx_p *= 2.0; Fz_p *= 2.0
        Fx_v *= 2.0; Fz_v *= 2.0
        My *= 2.0
        Fy_p = 0.0; Fy_v = 0.0
        Mx = 0.0; Mz = 0.0
    end

    force_data.Fx_pressure = Fx_p
    force_data.Fy_pressure = Fy_p
    force_data.Fz_pressure = Fz_p
    force_data.Fx_viscous = Fx_v
    force_data.Fy_viscous = Fy_v
    force_data.Fz_viscous = Fz_v

    force_data.Fx = Fx_p + Fx_v
    force_data.Fy = Fy_p + Fy_v
    force_data.Fz = Fz_p + Fz_v

    force_data.Mx = Mx
    force_data.My = My
    force_data.Mz = Mz

    q_inf = 0.5 * force_data.rho_ref * force_data.u_ref^2
    F_ref = q_inf * force_data.area_ref
    M_ref = F_ref * force_data.chord_ref

    if F_ref > 1e-10
        force_data.Cd = force_data.Fx / F_ref
        force_data.Cl = force_data.Fz / F_ref
        force_data.Cs = force_data.Fy / F_ref
    end

    if M_ref > 1e-10
        force_data.Cmx = force_data.Mx / M_ref
        force_data.Cmy = force_data.My / M_ref
        force_data.Cmz = force_data.Mz / M_ref
    end
end

function compute_aerodynamics!(force_data::ForceData, level, gpu_mesh, backend, params;
                               search_radius::Int=5)
    map_surface_stresses!(force_data, level, gpu_mesh, backend, params;
                          search_radius=search_radius)

    integrate_surface_forces!(force_data, gpu_mesh, backend, params)
end



function map_minus_side_stresses!(force_data::ForceData, level, gpu_mesh,
                                  backend, params; search_radius::Int=5)

    mesh_offset = params.mesh_offset
    tau_mol = level.tau

    velocity_scale = params.velocity_scale
    rho_phys = params.rho_physical

    pressure_scale = Float32(rho_phys * velocity_scale * velocity_scale)
    stress_scale = pressure_scale

    kernel! = map_minus_side_only_kernel!(backend)
    kernel!(
        force_data.pressure_map_minus,
        force_data.shear_x_map_minus, force_data.shear_y_map_minus, force_data.shear_z_map_minus,
        gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        level.rho, level.vel, level.obstacle, level.block_pointer,
        Int32(BLOCK_SIZE), Float32(level.dx),
        Int32(level.grid_dim_x), Int32(level.grid_dim_y), Int32(level.grid_dim_z),
        Float32(mesh_offset[1]), Float32(mesh_offset[2]), Float32(mesh_offset[3]),
        pressure_scale, stress_scale,
        Float32(tau_mol),
        Int32(search_radius),
        Float32(C_SMAGO),
        Float32(NU_SGS_BACKGROUND),
        ndrange=(gpu_mesh.n_triangles,)
    )

    KernelAbstractions.synchronize(backend)
end

function compute_net_triangle_forces!(force_data::ForceData, gpu_mesh, backend)
    n_tri = gpu_mesh.n_triangles

    net_Fx_gpu = KernelAbstractions.zeros(backend, Float32, n_tri)
    net_Fy_gpu = KernelAbstractions.zeros(backend, Float32, n_tri)
    net_Fz_gpu = KernelAbstractions.zeros(backend, Float32, n_tri)

    kernel! = compute_net_triangle_forces_kernel!(backend)
    kernel!(
        net_Fx_gpu, net_Fy_gpu, net_Fz_gpu,
        force_data.pressure_map,
        force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
        force_data.pressure_map_minus,
        force_data.shear_x_map_minus, force_data.shear_y_map_minus, force_data.shear_z_map_minus,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        gpu_mesh.areas,
        ndrange=(n_tri,)
    )

    KernelAbstractions.synchronize(backend)

    return (Array(net_Fx_gpu), Array(net_Fy_gpu), Array(net_Fz_gpu))
end

function compute_net_aerodynamics!(force_data::ForceData, level, gpu_mesh, backend, params;
                                   search_radius::Int=5, closed_body::Bool=true)
    # PLUS SIDE: reuse the proven one-sided mapping kernel (map_stresses_kernel!)
    # This populates force_data.pressure_map, shear_x/y/z_map with robust plus-side values
    map_surface_stresses!(force_data, level, gpu_mesh, backend, params;
                          search_radius=search_radius)

    if closed_body
        # For closed bodies (cube, car, aircraft fuselage, etc.):
        # No fluid exists inside the body, so minus-side contribution is exactly zero.
        # Net force = plus-side force directly.
        # This avoids spurious results from unmasked interior cells in block-structured grids.

        # Zero out minus maps (in case they held stale data)
        fill!(force_data.pressure_map_minus, 0.0f0)
        fill!(force_data.shear_x_map_minus, 0.0f0)
        fill!(force_data.shear_y_map_minus, 0.0f0)
        fill!(force_data.shear_z_map_minus, 0.0f0)

        println("[NetForce] Closed-body mode: skipping minus-side probing (net = plus-side)")

    else
        # MINUS SIDE: for thin/open surfaces where fluid exists on both sides
        map_minus_side_stresses!(force_data, level, gpu_mesh, backend, params;
                                 search_radius=search_radius)

        # --- Diagnostic: check minus-side coverage ---
        p_minus_cpu = Array(force_data.pressure_map_minus)
        n_total = gpu_mesh.n_triangles
        n_minus = count(x -> abs(x) > 1e-10, p_minus_cpu)
        println("[NetForce] Minus-side coverage: $(n_minus)/$(n_total) ($(round(100*n_minus/n_total, digits=1))%)")
        if n_minus > 0
            pm_nz = filter(x -> abs(x) > 1e-10, p_minus_cpu)
            @printf("[NetForce] Minus-side pressure range: [%.3e, %.3e] Pa\n", minimum(pm_nz), maximum(pm_nz))
        end
    end

    # Compute net force per triangle: F_net = F_plus - F_minus
    net_Fx, net_Fy, net_Fz = compute_net_triangle_forces!(force_data, gpu_mesh, backend)

    # --- Diagnostic: net force coverage ---
    p_plus_cpu = Array(force_data.pressure_map)
    n_total = gpu_mesh.n_triangles
    n_plus = count(x -> abs(x) > 1e-10, p_plus_cpu)
    net_mag = sqrt.(net_Fx.^2 .+ net_Fy.^2 .+ net_Fz.^2)
    n_nonzero_net = count(x -> x > 1e-10, net_mag)
    println("[NetForce] Plus-side  coverage: $(n_plus)/$(n_total) ($(round(100*n_plus/n_total, digits=1))%)")
    println("[NetForce] Net force coverage: $(n_nonzero_net)/$(n_total) ($(round(100*n_nonzero_net/n_total, digits=1))%)")
    if n_nonzero_net > 0
        nz_mag = filter(x -> x > 1e-10, net_mag)
        @printf("[NetForce] Net |F| range: [%.3e, %.3e] N\n", minimum(nz_mag), maximum(nz_mag))
    end
    # ---

    return (net_Fx, net_Fy, net_Fz)
end