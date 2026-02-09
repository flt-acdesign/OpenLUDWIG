ENV["LC_ALL"] = "C"

include("dependencies.jl")


const SIMULATION_START_TIME = Ref(time())

walltime_str() = begin
    e = time() - SIMULATION_START_TIME[]
    @sprintf("%02d:%02d:%05.2f", floor(Int, e/3600), floor(Int, (e%3600)/60), e%60)
end

log_walltime(msg::String) = println("[$(walltime_str())] $msg")

function force_cleanup()
    if CUDA.functional()
        GC.gc(true); GC.gc(true); CUDA.reclaim()
        free_mem, total_mem = CUDA.memory_info()
        @printf("[Memory] VRAM: %.2f GB free / %.2f GB total\n", free_mem/1024^3, total_mem/1024^3)
        return free_mem
    end
    return 0
end


function solve_main()
    SIMULATION_START_TIME[] = time()
    sim_start_datetime = Dates.now()

    println("\n" * "="^70)
    solver_mode = COMPRESSIBILITY_CORRECTION ? "D3Q27 | Compressible" : "D3Q27 | Incompressible"
    println("    LBM SOLVER | $(solver_mode) | WALE LES | SURFACE FORCE METHOD")
    println("    Case: $(basename(CASE_DIR)) | $(sim_start_datetime)")
    println("="^70)
    
    c_wale = haskey(CFG["advanced"]["numerics"], "c_wale") ? Float32(CFG["advanced"]["numerics"]["c_wale"]) : C_SMAGO
    inlet_turb = INLET_TURBULENCE_INTENSITY
    nu_sgs_bg = NU_SGS_BACKGROUND
    use_temporal = TEMPORAL_INTERPOLATION
    sponge_blend = SPONGE_BLEND_DISTRIBUTIONS
    
    println("\n[Config] Stability Settings:")
    @printf("          Background ν_sgs: %.6f → τ_eff_min ≈ %.4f\n", nu_sgs_bg, 0.5 + 3*nu_sgs_bg)
    println("          Sponge f-blending: $sponge_blend")
    println("          Temporal interpolation: $use_temporal")
    
    force_cleanup()
    
    gpu_device_name = CUDA.functional() ? string(CUDA.name(CUDA.device())) : "CPU"
    backend = CUDA.functional() ? (println("[Backend] CUDA: $(gpu_device_name)"); CUDABackend()) : (println("[Backend] CPU"); CPU())
    if CUDA.functional(); CUDA.allowscalar(false); end
    
    output_dir = OUT_DIR
    isdir(output_dir) ? (for f in readdir(output_dir); rm(joinpath(output_dir, f); recursive=true, force=true); end) : mkdir(output_dir)
    
    csv_path = joinpath(output_dir, "convergence.csv")
    open(csv_path, "w") do io; println(io, "Step,Time_phys_s,Domain_travel_pct,U_inlet_lat,Rho_min,MLUPS,Cd,Cl,Cs"); end
    
    force_csv = joinpath(output_dir, "forces.csv")
    if FORCE_COMPUTATION_ENABLED; write_force_csv_header(force_csv); end
    
    # Resolve STL paths (supports both single and multiple STL files)
    stl_paths = String[]
    for sf in STL_FILES
        isfile(sf) ? push!(stl_paths, sf) : error("STL file not found: $sf")
    end
    if isempty(stl_paths)
        fallback = joinpath(CASE_DIR, "model.stl")
        isfile(fallback) ? push!(stl_paths, fallback) : error("No STL files found")
    end

    log_walltime("Building domain...")
    cpu_grids, geometry_mesh, unrotated_mesh, unrotated_centers, object_names = setup_multilevel_domain(stl_paths; num_levels=NUM_LEVELS_CONFIG)
    
    params = get_domain_params()
    domain_nx, domain_ny, domain_nz = params.nx_coarse, params.ny_coarse, params.nz_coarse

    # ── Compute STEPS from domain_length_covered if specified ──
    if DOMAIN_LENGTH_COVERED > 0.0
        domain_x_phys = params.domain_size[1]
        physical_time_needed = DOMAIN_LENGTH_COVERED * domain_x_phys / params.u_physical
        dt_fine = params.time_scale
        computed_steps = Int(ceil(physical_time_needed / dt_fine))
        # Round up to nearest multiple of DIAG_FREQ for clean output
        computed_steps = Int(ceil(computed_steps / DIAG_FREQ)) * DIAG_FREQ
        global STEPS = computed_steps
        @printf("[Info] domain_length_covered = %.2f → physical time = %.4f s → STEPS = %d\n",
                DOMAIN_LENGTH_COVERED, physical_time_needed, STEPS)
    end
    if STEPS <= 0
        error("STEPS must be > 0. Set either 'steps' or 'domain_length_covered' in config.")
    end

    Ma_lattice = Float64(U_TARGET) * sqrt(3.0)
    Ma_physical = params.u_physical / 343.0
    @printf("[Info] Re = %.0f, Ma_phys = %.4f, Ma_lat = %.4f\n", params.re_number, Ma_physical, Ma_lattice)
    println("[Info] τ_levels = $(join([@sprintf("%.6f", t) for t in params.tau_levels], ", "))")
    if COMPRESSIBILITY_CORRECTION
        println("[Info] Compressible solver active: O(u⁴) equilibrium + thermal DDF")
    end
    
    log_walltime("Transferring to GPU...")
    grids = [adapt(backend, g) for g in cpu_grids]
    
    gpu_mesh = Geometry.upload_mesh_to_gpu(geometry_mesh, backend)
    
    cpu_grids = nothing; GC.gc()
    
    cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu = build_lattice_arrays_gpu(backend)

    log_walltime("Initializing equilibrium...")
    
    @kernel function init_eq!(f, f_temp, f_old, rho_old, vel_old, W, has_old::Int32)
        x, y, z, b = @index(Global, NTuple)
        @inbounds begin
            for k in 1:27
                f[x, y, z, b, k] = W[k]
                f_temp[x, y, z, b, k] = W[k]
            end
            if has_old == Int32(1)
                for k in 1:27; f_old[x, y, z, b, k] = W[k]; end
                rho_old[x, y, z, b] = 1.0f0
                vel_old[x, y, z, b, 1] = 0.0f0
                vel_old[x, y, z, b, 2] = 0.0f0
                vel_old[x, y, z, b, 3] = 0.0f0
            end
        end
    end

    # Thermal DDF initialization kernel (Phase 2)
    @kernel function init_thermal_eq!(g, g_temp, g_old, g_post_collision,
                                       temperature, temperature_old,
                                       W, T_init::Float32, has_old::Int32, has_bouzidi::Int32)
        x, y, z, b = @index(Global, NTuple)
        @inbounds begin
            for k in 1:27
                g_eq = W[k] * T_init
                g[x, y, z, b, k] = g_eq
                g_temp[x, y, z, b, k] = g_eq
            end
            temperature[x, y, z, b] = T_init
            temperature_old[x, y, z, b] = T_init
            if has_old == Int32(1)
                for k in 1:27; g_old[x, y, z, b, k] = W[k] * T_init; end
            end
            if has_bouzidi == Int32(1)
                for k in 1:27; g_post_collision[x, y, z, b, k] = W[k] * T_init; end
            end
        end
    end

    for lvl in 1:length(grids)
        level = grids[lvl]
        n = length(level.active_block_coords)
        if n > 0
            has_old = has_temporal_storage(level) ? Int32(1) : Int32(0)
            init_eq!(backend)(level.f, level.f_temp, level.f_old, level.rho_old, level.vel_old, w_gpu, has_old,
                              ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))

            # Initialize thermal DDF (Phase 2)
            if DDF_ENABLED && has_thermal(level)
                # Set tau_g on this level from domain params
                level.tau_g = params.tau_g_levels[lvl]
                has_old_g = (length(level.g_old) > 27) ? Int32(1) : Int32(0)
                has_bouz_g = (level.bouzidi_enabled && level.n_boundary_cells > 0) ? Int32(1) : Int32(0)
                init_thermal_eq!(backend)(level.g, level.g_temp, level.g_old, level.g_post_collision,
                                          level.temperature, level.temperature_old,
                                          w_gpu, Float32(DDF_T_INITIAL), has_old_g, has_bouz_g,
                                          ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
            end
        end
    end
    KernelAbstractions.synchronize(backend)
    
    total_cells = sum(length(g.active_block_coords) * BLOCK_SIZE^3 for g in grids)
    println("[Info] Total cells: $(round(total_cells/1e6, digits=2)) M")
    
    print_vram_breakdown(grids)
    
    force_data = nothing
    force_accm = nothing
    if FORCE_COMPUTATION_ENABLED
        force_data = ForceData(gpu_mesh.n_triangles, backend; 
                               rho_ref=params.rho_physical, 
                               u_ref=params.u_physical,
                               area_ref=params.reference_area, 
                               chord_ref=params.reference_chord,
                               moment_center=params.moment_center, 
                               force_scale=params.force_scale,
                               length_scale=params.length_scale, 
                               symmetric=SYMMETRIC_ANALYSIS)
        
        force_accm = ForceAccumulator(force_data)
        
        println("[Forces] Initialized surface stress method for $(gpu_mesh.n_triangles) triangles")
        println("          Reference: ρ=$(round(params.rho_physical, digits=3)) kg/m³, U=$(round(params.u_physical, digits=2)) m/s")
        println("          Reference: A=$(round(params.reference_area, digits=4)) m², L=$(round(params.reference_chord, digits=4)) m")
    end
    
    domain_length_x = params.domain_size[1]

    log_walltime("LBM Analysis STARTED")
    println()
    @printf("%8s | %10s | %7s | %7s | %7s | %6s | %8s | %8s | %8s\n",
            "Step", "Time[s]", "%Domain", "U_lat", "ρ_min", "MLUPS", "Cd", "Cl", "Cs")
    println(repeat("-", 100))

    t_start_sim = time()
    last_diag = time()
    batch = GPU_ASYNC_DEPTH
    
    t = 1
    while t <= STEPS
        batch_end = min(t + batch - 1, STEPS)
        actual = batch_end - t + 1
        
        prog = batch_end <= RAMP_STEPS ? 0.5f0 * (1.0f0 - cos(Float32(pi) * batch_end / RAMP_STEPS)) : 1.0f0
        u_curr = U_TARGET * prog
        
        execute_timestep_batch!(grids, t, actual, u_curr,
                                cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                domain_nx, domain_ny, domain_nz,
                                params.wall_model_active, c_wale, nu_sgs_bg,
                                inlet_turb, use_temporal, sponge_blend)
        
        if batch_end % DIAG_FREQ < actual || batch_end == STEPS
            diag_step = (batch_end ÷ DIAG_FREQ) * DIAG_FREQ
            if diag_step >= t && diag_step <= batch_end
                stats = compute_flow_stats(grids[1])
                
                now_t = time()
                mlups = (total_cells * DIAG_FREQ) / ((now_t - last_diag) * 1e6)
                last_diag = now_t
                time_phys = Float64(diag_step) * params.time_scale
                
                cd_str, cl_str, cs_str = "N/A", "N/A", "N/A"
                if force_data !== nothing
                    compute_aerodynamics!(force_data, grids[end], gpu_mesh, backend, params;
                                          search_radius=5)

                    if diag_step > RAMP_STEPS
                        accumulate_forces!(force_accm, force_data)
                    end

                    cd_str = @sprintf("%.4f", force_data.Cd)
                    cl_str = @sprintf("%.4f", force_data.Cl)
                    cs_str = @sprintf("%.4f", force_data.Cs)
                    append_force_csv(force_csv, diag_step, time_phys, force_data, u_curr)
                end

                domain_travel_pct = domain_length_x > 0 ? 100.0 * (time_phys * params.u_physical) / domain_length_x : 0.0

                @printf("%8d | %10.4f | %6.1f%% | %.4f | %.4f | %6.1f | %8s | %8s | %8s\n",
                        diag_step, time_phys, domain_travel_pct, u_curr, stats.rho_min, mlups, cd_str, cl_str, cs_str)
                open(csv_path, "a") do io
                    @printf(io, "%d,%.6e,%.6e,%.4f,%.6f,%.1f,%s,%s,%s\n",
                            diag_step, time_phys, domain_travel_pct, u_curr, stats.rho_min, mlups, cd_str, cl_str, cs_str)
                end
            end
        end
        
        if batch_end % OUTPUT_FREQ < actual
            out_step = (batch_end ÷ OUTPUT_FREQ) * OUTPUT_FREQ
            if out_step >= t && out_step <= batch_end
                export_merged_mesh_sync(out_step, grids, output_dir, backend)
                
                if force_data !== nothing
                    if out_step != ((out_step ÷ DIAG_FREQ) * DIAG_FREQ)
                        compute_aerodynamics!(force_data, grids[end], gpu_mesh, backend, params;
                                              search_radius=5)
                        
                        if out_step > RAMP_STEPS
                            accumulate_forces!(force_accm, force_data)
                        end
                    end
                    
                    surf_filename = @sprintf("%s/surface_%06d", output_dir, out_step)
                    save_surface_vtk(surf_filename, force_data, geometry_mesh, params;
                                     object_names=object_names)

                    # ── Binary surface pressure export (for web viewer) ──
                    # Uses unrotated mesh (STL frame, no domain offset) so that
                    # pressures and net force vectors share the same coordinate system.
                    surf_bin_filename = @sprintf("%s/surface_%06d.lbmp", output_dir, out_step)
                    export_surface_pressure_bin(surf_bin_filename, force_data, unrotated_mesh, params;
                                                object_names=object_names,
                                                use_domain_offset=false)

                    # ── Net surface force export (two-sided) ──
                    if NET_FORCE_EXPORT_ENABLED
                        net_Fx, net_Fy, net_Fz = compute_net_aerodynamics!(
                            force_data, grids[end], gpu_mesh, backend, params;
                            search_radius=5
                        )
                        export_net_triangle_forces_csv(
                            @sprintf("%s/net_forces_%06d.csv", output_dir, out_step),
                            net_Fx, net_Fy, net_Fz,
                            gpu_mesh, params.mesh_offset;
                            unrotated_centers=unrotated_centers,
                            alpha_deg=ALPHA_DEG,
                            beta_deg=BETA_DEG,
                            object_names=object_names
                        )
                        export_net_triangle_forces_vtk(
                            @sprintf("%s/net_forces_%06d", output_dir, out_step),
                            net_Fx, net_Fy, net_Fz,
                            gpu_mesh, params.mesh_offset
                        )
                    end
                end
            end
        end
        
        t = batch_end + 1
    end

    # ── Final output: ensure results are written even if STEPS < OUTPUT_FREQ ──
    # Check whether the last step already triggered an output
    last_output_step = (STEPS ÷ OUTPUT_FREQ) * OUTPUT_FREQ
    if last_output_step != STEPS || STEPS < OUTPUT_FREQ
        final_step = STEPS
        log_walltime("Writing final output at step $final_step...")

        try
            export_merged_mesh_sync(final_step, grids, output_dir, backend)
        catch e
            println("[Warning] Final VTK export failed: $e")
        end

        if force_data !== nothing
            # Ensure aerodynamics are computed for the final state
            compute_aerodynamics!(force_data, grids[end], gpu_mesh, backend, params;
                                  search_radius=5)

            try
                surf_filename = @sprintf("%s/surface_%06d", output_dir, final_step)
                save_surface_vtk(surf_filename, force_data, geometry_mesh, params;
                                 object_names=object_names)

                surf_bin_filename = @sprintf("%s/surface_%06d.lbmp", output_dir, final_step)
                export_surface_pressure_bin(surf_bin_filename, force_data, unrotated_mesh, params;
                                            object_names=object_names,
                                            use_domain_offset=false)
            catch e
                println("[Warning] Final surface export failed: $e")
            end

            if NET_FORCE_EXPORT_ENABLED
                try
                    net_Fx, net_Fy, net_Fz = compute_net_aerodynamics!(
                        force_data, grids[end], gpu_mesh, backend, params;
                        search_radius=5
                    )
                    export_net_triangle_forces_csv(
                        @sprintf("%s/net_forces_%06d.csv", output_dir, final_step),
                        net_Fx, net_Fy, net_Fz,
                        gpu_mesh, params.mesh_offset;
                        unrotated_centers=unrotated_centers,
                        alpha_deg=ALPHA_DEG,
                        beta_deg=BETA_DEG,
                        object_names=object_names
                    )
                    export_net_triangle_forces_vtk(
                        @sprintf("%s/net_forces_%06d", output_dir, final_step),
                        net_Fx, net_Fy, net_Fz,
                        gpu_mesh, params.mesh_offset
                    )
                catch e
                    println("[Warning] Final net forces export failed: $e")
                end
            end
        end
    end

    total_time = time() - t_start_sim
    println("\n" * "="^70)
    @printf("    SIMULATION COMPLETE | Wall time: %.1f s | Performance: %.1f MLUPS\n", total_time, (total_cells * STEPS) / (total_time * 1e6))
    println("="^70)
    
    if force_accm !== nothing
        print_averaged_summary(force_accm; re_number=params.re_number,
                               ma_physical=Ma_physical, ma_lattice=Ma_lattice,
                               compressibility_correction=COMPRESSIBILITY_CORRECTION)
    elseif force_data !== nothing
        print_force_summary(force_data; re_number=params.re_number,
                            ma_physical=Ma_physical, ma_lattice=Ma_lattice,
                            compressibility_correction=COMPRESSIBILITY_CORRECTION)
    end

    # ── Write comprehensive analysis summary ──
    sim_end_datetime = Dates.now()
    try
        write_analysis_summary(output_dir, params, force_data, force_accm,
                               total_cells, total_time,
                               sim_start_datetime, sim_end_datetime;
                               gpu_name=gpu_device_name)
    catch e
        println("[Warning] Failed to write analysis summary: $e")
    end

    # ── Build case result for multi-case summary ──
    case_result = build_case_result(params, force_data, force_accm,
                                     total_cells, total_time,
                                     sim_start_datetime, sim_end_datetime)

    grids = nothing
    force_cleanup()

    return case_result
end


function run_all_cases()
    cases_file = joinpath(@__DIR__, "../cases_to_run.yaml")
    if !isfile(cases_file); error("cases_to_run.yaml not found!"); end
    config = YAML.load_file(cases_file)
    cases = config["case_folders"]

    batch_start_datetime = Dates.now()
    case_results = []

    println("="^70)
    println("      MULTI-CASE EXECUTION: $(length(cases)) cases")
    println("="^70)
    for (i, case_name) in enumerate(cases)
        println("\n>>> CASE $i/$(length(cases)): $case_name")
        try
            load_case_configuration(case_name)
            result = solve_main()
            push!(case_results, result)
        catch e
            println("!!! ERROR: $e")
            showerror(stdout, e, catch_backtrace())
            # Record failed case so it appears in the summary
            push!(case_results, (
                case_name  = case_name,
                status     = :failed,
                error_msg  = string(e),
                re_number  = 0.0,
                u_physical = 0.0,
                alpha_deg  = 0.0,
                beta_deg   = 0.0,
                total_cells = 0,
                wall_time  = 0.0,
                mlups      = 0.0,
                steps      = 0,
                start_time = Dates.now(),
                end_time   = Dates.now(),
                # Final instantaneous
                Cd = 0.0, Cl = 0.0, Cs = 0.0,
                Cdp = 0.0, Cdv = 0.0,
                Cmx = 0.0, Cmy = 0.0, Cmz = 0.0,
                # Averaged
                avg_Cd = 0.0, avg_Cl = 0.0, avg_Cs = 0.0,
                avg_Cdp = 0.0, avg_Cdv = 0.0,
                avg_Cmx = 0.0, avg_Cmy = 0.0, avg_Cmz = 0.0,
                n_samples = 0,
                min_Cd = 0.0, max_Cd = 0.0,
                min_Cl = 0.0, max_Cl = 0.0,
                min_Cs = 0.0, max_Cs = 0.0,
                collision_operator = :unknown,
                ma_physical = 0.0,
                compressibility_correction = false,
                output_dir = ""
            ))
        end
        GC.gc(true)
        if CUDA.functional(); CUDA.reclaim(); end
    end

    batch_end_datetime = Dates.now()
    println("\n" * "="^70 * "\n      ALL CASES COMPLETED\n" * "="^70)

    # ── Write multi-case summary ──
    cases_dir = dirname(cases_file)
    try
        write_cases_summary(cases_dir, case_results, batch_start_datetime, batch_end_datetime)
    catch e
        println("[Warning] Failed to write cases summary: $e")
        showerror(stdout, e, catch_backtrace())
    end
end

run_all_cases()