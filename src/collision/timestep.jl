function perform_timestep_v2!(
    level,
    parent_f, parent_rho, parent_vel, parent_ptr,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_tau::Float32,
    f_out, f_in, vel_out, vel_in,
    f_post_collision, u_curr,
    cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
    domain_nx, domain_ny, domain_nz,
    wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
    timestep::Int, inlet_turbulence::Float32,
    temporal_weight::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool;
    # Thermal DDF (Phase 2) — optional keyword arguments
    g_out=nothing, g_in=nothing, g_post_collision_arr=nothing,
    parent_g=nothing, parent_temperature=nothing,
    parent_g_old=nothing, parent_temperature_old=nothing,
    parent_vel_thermal=nothing, parent_vel_old_thermal=nothing,
    parent_tau_g::Float32=0.5f0
)
    backend = get_backend(f_in)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0; return; end
    
    is_l1 = (parent_f === nothing)
    
    if is_l1
        p_f = f_in; p_rho = level.rho; p_vel = level.vel; p_ptr = level.block_pointer
        p_f_old = f_in; p_rho_old = level.rho; p_vel_old = level.vel
        px, py, pz = Int32(1), Int32(1), Int32(1)
    else
        p_f = parent_f; p_rho = parent_rho; p_vel = parent_vel; p_ptr = parent_ptr
        p_f_old = parent_f_old; p_rho_old = parent_rho_old; p_vel_old = parent_vel_old
        px, py, pz = Int32(size(p_ptr, 1)), Int32(size(p_ptr, 2)), Int32(size(p_ptr, 3))
    end
    
    scale = 1 << (level.level_id - 1)
    nx_g, ny_g, nz_g = Int32(domain_nx * scale), Int32(domain_ny * scale), Int32(domain_nz * scale)
    
    # Phase 2: Read cumulant parameters from globals
    omega_bulk_p  = CUMULANT_OMEGA_BULK
    omega_3_p     = CUMULANT_OMEGA_3
    omega_4_p     = CUMULANT_OMEGA_4
    adaptive_om4  = CUMULANT_ADAPTIVE_OMEGA4 ? Int32(1) : Int32(0)
    lambda_p      = CUMULANT_LAMBDA_PARAM
    # Encode limiter: 0=none, 1=factored, 2=positivity
    limiter_i     = CUMULANT_LIMITER == :factored ? Int32(1) :
                    CUMULANT_LIMITER == :positivity ? Int32(2) : Int32(0)
    # Compressibility correction flag
    comp_corr_i   = COMPRESSIBILITY_CORRECTION ? Int32(1) : Int32(0)
    
    kernel! = get_collision_kernel(COLLISION_OPERATOR, backend)
    
    kernel!(
        f_out, f_in, f_post_collision,
        level.rho, vel_out, vel_in,
        level.obstacle, level.sponge, level.wall_dist,
        level.neighbor_table,
        level.map_x, level.map_y, level.map_z,
        p_f, p_rho, p_vel, p_ptr,
        p_f_old, p_rho_old, p_vel_old,
        px, py, pz,
        level.tau, parent_tau, c_wale_val, nu_sgs_bg,
        CS2_PHYSICS, CS4_PHYSICS,
        is_l1 ? Int32(1) : Int32(0),
        SYMMETRIC_ANALYSIS ? Int32(1) : Int32(0),
        nx_g, ny_g, nz_g, u_curr,
        Int32(n_blocks), Int32(BLOCK_SIZE),
        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
        wall_model_active ? Int32(1) : Int32(0),
        Int32(timestep % 1000000),
        (level.bouzidi_enabled && level.n_boundary_cells > 0) ? Int32(1) : Int32(0),
        inlet_turbulence,
        Float32(temporal_weight),
        use_temporal_interp ? Int32(1) : Int32(0),
        sponge_blend_dist ? Int32(1) : Int32(0),
        # Phase 2: new cumulant parameters
        omega_bulk_p,
        omega_3_p,
        omega_4_p,
        adaptive_om4,
        lambda_p,
        limiter_i,
        # Compressibility correction
        comp_corr_i,
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )
    
    KernelAbstractions.synchronize(backend)
    
    if level.bouzidi_enabled && level.n_boundary_cells > 0
        apply_bouzidi_correction!(
            f_out, f_post_collision,
            level.bouzidi_q_map, level.bouzidi_cell_block,
            level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
            level.n_boundary_cells, level.neighbor_table, BLOCK_SIZE,
            cx_gpu, cy_gpu, cz_gpu, opp_gpu, Q_MIN_THRESHOLD, backend
        )
        KernelAbstractions.synchronize(backend)
    end

    # ── Thermal DDF kernel (Phase 2) ──
    if DDF_ENABLED && g_out !== nothing && g_in !== nothing && has_thermal(level)
        # Parent thermal data
        if is_l1
            pg = g_in; pT = level.temperature; pvel_t = level.vel; pptr_t = level.block_pointer
            pg_old = g_in; pT_old = level.temperature; pvel_old_t = level.vel
            pgx, pgy, pgz = Int32(1), Int32(1), Int32(1)
        else
            pg = parent_g; pT = parent_temperature; pvel_t = parent_vel_thermal !== nothing ? parent_vel_thermal : p_vel
            pg_old = parent_g_old; pT_old = parent_temperature_old
            pvel_old_t = parent_vel_old_thermal !== nothing ? parent_vel_old_thermal : p_vel_old
            pptr_t = p_ptr
            pgx, pgy, pgz = px, py, pz
        end

        # Thermal wall BC flag
        wall_bc_adiabatic_i = DDF_WALL_BC == :adiabatic ? Int32(1) : Int32(0)

        thermal_kernel! = stream_collide_thermal!(backend)
        thermal_kernel!(
            g_out, g_in, g_post_collision_arr !== nothing ? g_post_collision_arr : level.g_post_collision,
            level.temperature, level.rho, vel_out,
            level.obstacle, level.sponge,
            level.neighbor_table,
            level.map_x, level.map_y, level.map_z,
            pg, pT, pvel_t, pptr_t,
            pg_old, pT_old, pvel_old_t,
            pgx, pgy, pgz,
            level.tau_g,
            parent_tau_g,
            is_l1 ? Int32(1) : Int32(0),
            SYMMETRIC_ANALYSIS ? Int32(1) : Int32(0),
            nx_g, ny_g, nz_g, u_curr,
            Float32(DDF_T_INLET),
            Float32(DDF_T_WALL),
            wall_bc_adiabatic_i,
            Int32(n_blocks), Int32(BLOCK_SIZE),
            cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
            (level.bouzidi_enabled && level.n_boundary_cells > 0) ? Int32(1) : Int32(0),
            Float32(temporal_weight),
            use_temporal_interp ? Int32(1) : Int32(0),
            sponge_blend_dist ? Int32(1) : Int32(0),
            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        )
        KernelAbstractions.synchronize(backend)

        # Bouzidi for thermal distributions
        if level.bouzidi_enabled && level.n_boundary_cells > 0
            apply_bouzidi_correction!(
                g_out, g_post_collision_arr !== nothing ? g_post_collision_arr : level.g_post_collision,
                level.bouzidi_q_map, level.bouzidi_cell_block,
                level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
                level.n_boundary_cells, level.neighbor_table, BLOCK_SIZE,
                cx_gpu, cy_gpu, cz_gpu, opp_gpu, Q_MIN_THRESHOLD, backend
            )
            KernelAbstractions.synchronize(backend)
        end
    end
end

function build_lattice_arrays_gpu(backend)
    cx, cy, cz, w = Int32[], Int32[], Int32[], Float32[]
    
    for dz in -1:1, dy in -1:1, dx in -1:1
        push!(cx, Int32(dx)); push!(cy, Int32(dy)); push!(cz, Int32(dz))
        d2 = dx^2 + dy^2 + dz^2
        push!(w, d2==0 ? 8f0/27f0 : d2==1 ? 2f0/27f0 : d2==2 ? 1f0/54f0 : 1f0/216f0)
    end
    
    opp, mirror_y, mirror_z = zeros(Int32, 27), zeros(Int32, 27), zeros(Int32, 27)
    for i in 1:27, j in 1:27
        if cx[j]==-cx[i] && cy[j]==-cy[i] && cz[j]==-cz[i]; opp[i] = Int32(j); end
        if cx[j]==cx[i] && cy[j]==-cy[i] && cz[j]==cz[i]; mirror_y[i] = Int32(j); end
        if cx[j]==cx[i] && cy[j]==cy[i] && cz[j]==-cz[i]; mirror_z[i] = Int32(j); end
    end
    
    return (adapt(backend, cx), adapt(backend, cy), adapt(backend, cz), adapt(backend, w),
            adapt(backend, opp), adapt(backend, mirror_y), adapt(backend, mirror_z))
end