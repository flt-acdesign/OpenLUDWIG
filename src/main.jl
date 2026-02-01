"""
MAIN.JL - LBM Solver Entry Point

D3Q27 Lattice Boltzmann solver with:
- WALE turbulence model
- Multi-grid refinement
- Bouzidi interpolated bounce-back boundaries
- Surface-based aerodynamic force computation
"""

ENV["LC_ALL"] = "C"

using LinearAlgebra, Printf, WriteVTK, Dates, StaticArrays
using KernelAbstractions, CUDA, Adapt, Base.Threads, YAML

include("initialize.jl")
include("lattice.jl")

if !isfile(joinpath(@__DIR__, "geometry.jl"))
    error("geometry.jl not found")
end
include("geometry.jl")
using .Geometry

include("blocks.jl")
include("bouzidi.jl")
include("domain.jl")
include("physics_v2.jl")
include("diagnostics.jl")
include("forces.jl")
include("diagnostics_vram.jl")
include("io_vtk.jl")
include("solver_control.jl")

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
    
    println("\n" * "="^70)
    println("    LBM SOLVER | D3Q27 | WALE LES | SURFACE FORCE METHOD")
    println("    Case: $(basename(CASE_DIR)) | $(Dates.now())")
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
    
    backend = CUDA.functional() ? (println("[Backend] CUDA: $(CUDA.name(CUDA.device()))"); CUDABackend()) : (println("[Backend] CPU"); CPU())
    if CUDA.functional(); CUDA.allowscalar(false); end
    
    output_dir = OUT_DIR
    isdir(output_dir) ? (for f in readdir(output_dir); rm(joinpath(output_dir, f); recursive=true, force=true); end) : mkdir(output_dir)
    
    csv_path = joinpath(output_dir, "convergence.csv")
    open(csv_path, "w") do io; println(io, "Step,Walltime,Time_phys_s,U_inlet_lat,Rho_min,MLUPS,Cd,Cl"); end
    
    force_csv = joinpath(output_dir, "forces.csv")
    if FORCE_COMPUTATION_ENABLED; write_force_csv_header(force_csv); end
    
    stl_path = isfile(STL_FILE) ? STL_FILE : (isfile(joinpath(CASE_DIR, "model.stl")) ? joinpath(CASE_DIR, "model.stl") : error("STL not found"))
    
    log_walltime("Building domain...")
    cpu_grids, geometry_mesh = setup_multilevel_domain(stl_path; num_levels=NUM_LEVELS_CONFIG)
    
    params = get_domain_params()
    domain_nx, domain_ny, domain_nz = params.nx_coarse, params.ny_coarse, params.nz_coarse
    
    println("[Info] Re = $(round(params.re_number)), τ_levels = $(join([@sprintf("%.6f", t) for t in params.tau_levels], ", "))")
    
    log_walltime("Transferring to GPU...")
    grids = [adapt(backend, g) for g in cpu_grids]
    
    # Upload mesh geometry to GPU for force computation
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
    
    for lvl in 1:length(grids)
        level = grids[lvl]
        n = length(level.active_block_coords)
        if n > 0
            has_old = has_temporal_storage(level) ? Int32(1) : Int32(0)
            init_eq!(backend)(level.f, level.f_temp, level.f_old, level.rho_old, level.vel_old, w_gpu, has_old,
                              ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
        end
    end
    KernelAbstractions.synchronize(backend)
    
    total_cells = sum(length(g.active_block_coords) * BLOCK_SIZE^3 for g in grids)
    println("[Info] Total cells: $(round(total_cells/1e6, digits=2)) M")
    
    print_vram_breakdown(grids)
    
    # Initialize force computation (using surface stress integration method)
    force_data = nothing
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
        println("[Forces] Initialized surface stress method for $(gpu_mesh.n_triangles) triangles")
        println("         Reference: ρ=$(round(params.rho_physical, digits=3)) kg/m³, U=$(round(params.u_physical, digits=2)) m/s")
        println("         Reference: A=$(round(params.reference_area, digits=4)) m², L=$(round(params.reference_chord, digits=4)) m")
    end
    
    log_walltime("LBM Analysis STARTED")
    println()
    @printf("%8s | %12s | %10s | %7s | %7s | %6s | %8s | %8s\n", "Step", "Walltime", "Time[s]", "U_lat", "ρ_min", "MLUPS", "Cd", "Cl")
    println(repeat("-", 90))

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
        
        # Diagnostics output
        if batch_end % DIAG_FREQ < actual || batch_end == STEPS
            diag_step = (batch_end ÷ DIAG_FREQ) * DIAG_FREQ
            if diag_step >= t && diag_step <= batch_end
                stats = compute_flow_stats(grids[1])
                
                now_t = time()
                mlups = (total_cells * DIAG_FREQ) / ((now_t - last_diag) * 1e6)
                last_diag = now_t
                time_phys = Float64(diag_step) * params.time_scale
                
                cd_str, cl_str = "N/A", "N/A"
                if force_data !== nothing
                    # Compute aerodynamics using surface stress integration
                    # This maps pressure/shear to triangles and integrates
                    compute_aerodynamics!(force_data, grids[end], gpu_mesh, backend, params;
                                          search_radius=5)
                    
                    cd_str = @sprintf("%.4f", force_data.Cd)
                    cl_str = @sprintf("%.4f", force_data.Cl)
                    append_force_csv(force_csv, diag_step, time_phys, force_data, u_curr)
                end
                
                @printf("%8d | %12s | %10.4f | %.4f | %.4f | %6.1f | %8s | %8s\n",
                        diag_step, walltime_str(), time_phys, u_curr, stats.rho_min, mlups, cd_str, cl_str)
                open(csv_path, "a") do io
                    println(io, "$diag_step,$(walltime_str()),$time_phys,$u_curr,$(stats.rho_min),$mlups,$cd_str,$cl_str")
                end
            end
        end
        
        # VTK output
        if batch_end % OUTPUT_FREQ < actual
            out_step = (batch_end ÷ OUTPUT_FREQ) * OUTPUT_FREQ
            if out_step >= t && out_step <= batch_end
                export_merged_mesh_sync(out_step, grids, output_dir, backend)
                
                if force_data !== nothing
                    # Ensure stresses are mapped (they should be from diagnostics, but re-map for VTK step)
                    if out_step != ((out_step ÷ DIAG_FREQ) * DIAG_FREQ)
                        # VTK step doesn't coincide with diag step, need to compute
                        compute_aerodynamics!(force_data, grids[end], gpu_mesh, backend, params;
                                              search_radius=5)
                    end
                    
                    surf_filename = @sprintf("%s/surface_%06d", output_dir, out_step)
                    save_surface_vtk(surf_filename, force_data, geometry_mesh)
                end
            end
        end
        
        t = batch_end + 1
    end
    
    # Final summary
    total_time = time() - t_start_sim
    println("\n" * "="^70)
    @printf("    SIMULATION COMPLETE | Wall time: %.1f s | Performance: %.1f MLUPS\n", total_time, (total_cells * STEPS) / (total_time * 1e6))
    println("="^70)
    
    # Print final force summary
    if force_data !== nothing
        print_force_summary(force_data)
    end

    grids = nothing
    force_cleanup()
end

function run_all_cases()
    cases_file = joinpath(@__DIR__, "../cases_to_run.yaml")
    if !isfile(cases_file); error("cases_to_run.yaml not found!"); end
    config = YAML.load_file(cases_file)
    cases = config["case_folders"]
    println("="^70)
    println("      MULTI-CASE EXECUTION: $(length(cases)) cases")
    println("="^70)
    for (i, case_name) in enumerate(cases)
        println("\n>>> CASE $i/$(length(cases)): $case_name")
        try
            load_case_configuration(case_name)
            solve_main()
        catch e
            println("!!! ERROR: $e")
            showerror(stdout, e, catch_backtrace())
        end
        GC.gc(true)
        if CUDA.functional(); CUDA.reclaim(); end
    end
    println("\n" * "="^70 * "\n      ALL CASES COMPLETED\n" * "="^70)
end

run_all_cases()