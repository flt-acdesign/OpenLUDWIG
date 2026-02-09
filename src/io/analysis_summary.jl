# FILE: src/io/analysis_summary.jl
# Writes a comprehensive Analysis_summary.md file at the end of each simulation.

using Printf
using Dates

function write_analysis_summary(output_dir::String, params, force_data, force_accm,
                                 total_cells::Int, total_time::Float64,
                                 start_datetime::DateTime, end_datetime::DateTime;
                                 gpu_name::String="N/A")

    summary_path = joinpath(output_dir, "Analysis_summary.md")

    # ----- Pre-compute derived quantities -----
    q_inf = 0.5 * params.rho_physical * params.u_physical^2
    F_ref = q_inf * params.reference_area
    M_ref = F_ref * params.reference_chord
    Ma_lattice = Float64(U_TARGET) * sqrt(3.0)
    Ma_physical = params.u_physical / 343.0   # speed of sound in air ~ 343 m/s
    avg_mlups = (total_cells * STEPS) / (total_time * 1e6)

    # Config file path
    config_path = joinpath(CASE_DIR, "config.yaml")
    stl_path = STL_FILE

    open(summary_path, "w") do io

        # ══════════════════════════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════════════════════════
        println(io, "# Analysis Summary — LUDWIG v1.10")
        println(io, "")
        println(io, "## Run Information")
        println(io, "")
        println(io, "| Field | Value |")
        println(io, "|-------|-------|")
        @printf(io, "| **Case name** | `%s` |\n", basename(CASE_DIR))
        @printf(io, "| **Start time** | %s |\n", Dates.format(start_datetime, "yyyy-mm-dd HH:MM:SS"))
        @printf(io, "| **End time** | %s |\n", Dates.format(end_datetime, "yyyy-mm-dd HH:MM:SS"))
        @printf(io, "| **Wall time** | %.1f s (%.2f min) |\n", total_time, total_time / 60.0)
        @printf(io, "| **GPU** | %s |\n", gpu_name)
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # FILE PATHS
        # ══════════════════════════════════════════════════════════════
        println(io, "## File Paths")
        println(io, "")
        println(io, "| File | Path |")
        println(io, "|------|------|")
        @printf(io, "| **Case directory** | `%s` |\n", abspath(CASE_DIR))
        @printf(io, "| **Configuration** | `%s` |\n", abspath(config_path))
        @printf(io, "| **STL geometry** | `%s` |\n", abspath(stl_path))
        @printf(io, "| **Output directory** | `%s` |\n", abspath(output_dir))
        @printf(io, "| **Force CSV** | `%s` |\n", abspath(joinpath(output_dir, "forces.csv")))
        @printf(io, "| **Convergence CSV** | `%s` |\n", abspath(joinpath(output_dir, "convergence.csv")))
        @printf(io, "| **This summary** | `%s` |\n", abspath(summary_path))
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # GEOMETRY & MESH
        # ══════════════════════════════════════════════════════════════
        println(io, "## Geometry & Mesh")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| **STL file** | `%s` |\n", STL_FILENAME)
        @printf(io, "| **STL scale** | %.4f |\n", STL_SCALE)
        @printf(io, "| **Alpha (angle of attack)** | %.2f deg |\n", ALPHA_DEG)
        @printf(io, "| **Beta (sideslip)** | %.2f deg |\n", BETA_DEG)
        @printf(io, "| **Symmetric analysis** | %s |\n", SYMMETRIC_ANALYSIS ? "Yes" : "No")
        @printf(io, "| **Mesh bounding box min** | (%.4f, %.4f, %.4f) m |\n",
                params.mesh_min[1], params.mesh_min[2], params.mesh_min[3])
        @printf(io, "| **Mesh bounding box max** | (%.4f, %.4f, %.4f) m |\n",
                params.mesh_max[1], params.mesh_max[2], params.mesh_max[3])
        @printf(io, "| **Mesh extent** | (%.4f, %.4f, %.4f) m |\n",
                params.mesh_extent[1], params.mesh_extent[2], params.mesh_extent[3])
        @printf(io, "| **Mesh center** | (%.4f, %.4f, %.4f) m |\n",
                params.mesh_center[1], params.mesh_center[2], params.mesh_center[3])
        @printf(io, "| **Surface resolution** | %d cells/L |\n", SURFACE_RESOLUTION)
        if MINIMUM_FACET_SIZE > 0.0
            @printf(io, "| **Minimum facet size** | %.6f m |\n", MINIMUM_FACET_SIZE)
        else
            println(io, "| **Minimum facet size** | disabled |")
        end
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # REFERENCE VALUES
        # ══════════════════════════════════════════════════════════════
        println(io, "## Reference Values")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| **Reference length (meshing)** | %.4f m |\n", params.reference_length)
        @printf(io, "| **Reference chord** | %.4f m |\n", params.reference_chord)
        @printf(io, "| **Reference area** | %.4f m² |\n", params.reference_area)
        @printf(io, "| **Reference area (full model)** | %.4f m² |\n", REFERENCE_AREA_FULL_MODEL)
        @printf(io, "| **Reference dimension** | %s |\n", REFERENCE_DIMENSION)
        @printf(io, "| **Moment center** | (%.4f, %.4f, %.4f) m |\n",
                params.moment_center[1], params.moment_center[2], params.moment_center[3])
        @printf(io, "| **Dynamic pressure q∞** | %.4f Pa |\n", q_inf)
        @printf(io, "| **Force reference F_ref** | %.4f N |\n", F_ref)
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # FLOW CONDITIONS
        # ══════════════════════════════════════════════════════════════
        println(io, "## Flow Conditions")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| **Reynolds number** | %.2e |\n", params.re_number)
        @printf(io, "| **Velocity** | %.4f m/s |\n", params.u_physical)
        @printf(io, "| **Density** | %.4f kg/m³ |\n", params.rho_physical)
        @printf(io, "| **Kinematic viscosity** | %.4e m²/s |\n", params.nu_physical)
        @printf(io, "| **Mach number (physical)** | %.4f |\n", Ma_physical)
        @printf(io, "| **Mach number (lattice)** | %.4f |\n", Ma_lattice)
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # DOMAIN
        # ══════════════════════════════════════════════════════════════
        println(io, "## Domain Configuration")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| **Domain size** | (%.4f, %.4f, %.4f) m |\n",
                params.domain_size[1], params.domain_size[2], params.domain_size[3])
        @printf(io, "| **Upstream** | %.2f L_ref |\n", DOMAIN_UPSTREAM)
        @printf(io, "| **Downstream** | %.2f L_ref |\n", DOMAIN_DOWNSTREAM)
        @printf(io, "| **Lateral** | %.2f L_ref |\n", DOMAIN_LATERAL)
        @printf(io, "| **Height** | %.2f L_ref |\n", DOMAIN_HEIGHT)
        @printf(io, "| **Coarse grid** | %d × %d × %d cells |\n",
                params.nx_coarse, params.ny_coarse, params.nz_coarse)
        @printf(io, "| **Block grid** | %d × %d × %d blocks |\n",
                params.bx_max, params.by_max, params.bz_max)
        @printf(io, "| **Number of levels** | %d |\n", params.num_levels)
        @printf(io, "| **dx (fine)** | %.6f m |\n", params.dx_fine)
        @printf(io, "| **dx (coarse)** | %.6f m |\n", params.dx_coarse)
        @printf(io, "| **Mesh offset** | (%.4f, %.4f, %.4f) m |\n",
                params.mesh_offset[1], params.mesh_offset[2], params.mesh_offset[3])
        @printf(io, "| **Sponge thickness** | %.2f L_ref |\n", SPONGE_THICKNESS)
        @printf(io, "| **Block size** | %d |\n", BLOCK_SIZE_CONFIG)
        @printf(io, "| **Refinement margin** | %d |\n", REFINEMENT_MARGIN)
        @printf(io, "| **Refinement strategy** | %s |\n", REFINEMENT_STRATEGY)
        @printf(io, "| **Wake refinement** | %s |\n", ENABLE_WAKE_REFINEMENT ? "enabled" : "disabled")
        if ENABLE_WAKE_REFINEMENT
            @printf(io, "| **Wake length** | %.2f L_ref |\n", WAKE_REFINEMENT_LENGTH)
            @printf(io, "| **Wake width factor** | %.2f |\n", WAKE_REFINEMENT_WIDTH_FACTOR)
            @printf(io, "| **Wake height factor** | %.2f |\n", WAKE_REFINEMENT_HEIGHT_FACTOR)
        end
        @printf(io, "| **Total cells** | %.2f M |\n", total_cells / 1e6)
        @printf(io, "| **Estimated memory** | %.2f GB |\n", params.estimated_memory_gb)
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # NUMERICAL PARAMETERS
        # ══════════════════════════════════════════════════════════════
        println(io, "## Numerical Parameters")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| **Collision operator** | %s |\n", COLLISION_OPERATOR)
        @printf(io, "| **U_lattice (target)** | %.4f |\n", U_TARGET)
        @printf(io, "| **TAU_min** | %.6f |\n", TAU_MIN)
        @printf(io, "| **TAU safety factor** | %.2f |\n", TAU_SAFETY_FACTOR)
        println(io, "| **tau per level** | $(join([@sprintf("L%d: %.6f", i, params.tau_levels[i]) for i in 1:length(params.tau_levels)], ", ")) |")
        @printf(io, "| **tau (fine)** | %.6f |\n", params.tau_fine)
        @printf(io, "| **tau margin** | %.2f%% |\n", params.tau_margin_percent)
        @printf(io, "| **nu (lattice, fine)** | %.6e |\n", params.nu_lattice)
        @printf(io, "| **C_WALE** | %.4f |\n", C_SMAGO)
        @printf(io, "| **NU_SGS_background** | %.6f |\n", NU_SGS_BACKGROUND)
        @printf(io, "| **Inlet turbulence intensity** | %.4f |\n", INLET_TURBULENCE_INTENSITY)
        @printf(io, "| **Sponge f-blending** | %s |\n", SPONGE_BLEND_DISTRIBUTIONS ? "enabled" : "disabled")
        @printf(io, "| **Temporal interpolation** | %s |\n", TEMPORAL_INTERPOLATION ? "enabled" : "disabled")
        println(io, "")

        if COLLISION_OPERATOR == :cumulant
            println(io, "### Cumulant Operator Settings")
            println(io, "")
            println(io, "| Parameter | Value |")
            println(io, "|-----------|-------|")
            @printf(io, "| **omega_bulk** | %.4f |\n", CUMULANT_OMEGA_BULK)
            @printf(io, "| **omega_3** | %.4f |\n", CUMULANT_OMEGA_3)
            @printf(io, "| **omega_4** | %.4f |\n", CUMULANT_OMEGA_4)
            @printf(io, "| **Adaptive omega_4** | %s |\n", CUMULANT_ADAPTIVE_OMEGA4 ? "enabled" : "disabled")
            if CUMULANT_ADAPTIVE_OMEGA4
                @printf(io, "| **Lambda parameter (Λ)** | %.4f |\n", CUMULANT_LAMBDA_PARAM)
            end
            @printf(io, "| **Limiter** | %s |\n", CUMULANT_LIMITER)
            @printf(io, "| **Compressibility correction** | %s |\n", COMPRESSIBILITY_CORRECTION ? "enabled (O(u⁴) equilibrium)" : "disabled (standard O(u²))")
            println(io, "")
        end

        # ══════════════════════════════════════════════════════════════
        # THERMAL DDF (Phase 2 — auto-enabled by compressibility_correction)
        # ══════════════════════════════════════════════════════════════
        if DDF_ENABLED
            println(io, "## Thermal DDF Configuration (auto-enabled by compressibility correction)")
            println(io, "")
            println(io, "| Parameter | Value |")
            println(io, "|-----------|-------|")
            @printf(io, "| **Prandtl number** | %.3f |\n", DDF_PRANDTL)
            @printf(io, "| **T_inlet** | %.3f |\n", DDF_T_INLET)
            @printf(io, "| **T_wall** | %.3f |\n", DDF_T_WALL)
            @printf(io, "| **T_initial** | %.3f |\n", DDF_T_INITIAL)
            @printf(io, "| **Wall BC** | %s |\n", DDF_WALL_BC)
            if length(params.tau_g_levels) > 0
                @printf(io, "| **τ_g_fine** | %.6f |\n", params.tau_g_fine)
                @printf(io, "| **τ_g levels** | %s |\n", join([@sprintf("%.6f", t) for t in params.tau_g_levels], ", "))
            end
            println(io, "")
        end

        # ══════════════════════════════════════════════════════════════
        # BOUNDARY CONDITIONS
        # ══════════════════════════════════════════════════════════════
        println(io, "## Boundary Conditions")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| **Boundary method** | %s |\n", BOUNDARY_METHOD)
        @printf(io, "| **Bouzidi levels** | %d |\n", BOUZIDI_LEVELS)
        @printf(io, "| **Q min threshold** | %.4f |\n", Q_MIN_THRESHOLD)
        @printf(io, "| **Wall model** | %s |\n", WALL_MODEL_ENABLED ? "enabled" : "disabled")
        if WALL_MODEL_ENABLED
            @printf(io, "| **Wall model type** | %s |\n", WALL_MODEL_TYPE)
            @printf(io, "| **y+ target** | %.1f |\n", WALL_MODEL_YPLUS_TARGET)
        end
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # SIMULATION PARAMETERS
        # ══════════════════════════════════════════════════════════════
        println(io, "## Simulation Parameters")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        if DOMAIN_LENGTH_COVERED > 0.0
            @printf(io, "| **Domain length covered** | %.2f |\n", DOMAIN_LENGTH_COVERED)
        end
        @printf(io, "| **Total steps** | %d |\n", STEPS)
        @printf(io, "| **Ramp steps** | %d |\n", RAMP_STEPS)
        @printf(io, "| **Output frequency** | %d |\n", OUTPUT_FREQ)
        @printf(io, "| **Diagnostics frequency** | %d |\n", DIAG_FREQ)
        @printf(io, "| **GPU async depth** | %d |\n", GPU_ASYNC_DEPTH)
        @printf(io, "| **Physical time simulated** | %.6e s |\n", Float64(STEPS) * params.time_scale)
        @printf(io, "| **Physical time step (fine)** | %.6e s |\n", params.time_scale)
        domain_traversals = (Float64(STEPS) * params.time_scale * params.u_physical) / params.domain_size[1]
        @printf(io, "| **Domain traversals** | %.2f |\n", domain_traversals)
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # SCALING FACTORS
        # ══════════════════════════════════════════════════════════════
        println(io, "## Scaling Factors (LBM ↔ Physical)")
        println(io, "")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        @printf(io, "| **Length scale** | %.6e m/lu |\n", params.length_scale)
        @printf(io, "| **Time scale** | %.6e s/ts |\n", params.time_scale)
        @printf(io, "| **Velocity scale** | %.4f (m/s)/lu |\n", params.velocity_scale)
        @printf(io, "| **Force scale** | %.6e N/fu |\n", params.force_scale)
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # FORCE RESULTS
        # ══════════════════════════════════════════════════════════════
        if force_data !== nothing
            println(io, "## Aerodynamic Forces — Final Instantaneous")
            println(io, "")

            Cdp_final = F_ref > 1e-10 ? force_data.Fx_pressure / F_ref : 0.0
            Cdv_final = F_ref > 1e-10 ? force_data.Fx_viscous / F_ref : 0.0

            println(io, "### Forces [N]")
            println(io, "")
            println(io, "| Component | Total | Pressure | Viscous |")
            println(io, "|-----------|-------|----------|---------|")
            @printf(io, "| **Fx (drag)** | %+.4e | %+.4e | %+.4e |\n",
                    force_data.Fx, force_data.Fx_pressure, force_data.Fx_viscous)
            @printf(io, "| **Fy (side)** | %+.4e | %+.4e | %+.4e |\n",
                    force_data.Fy, force_data.Fy_pressure, force_data.Fy_viscous)
            @printf(io, "| **Fz (lift)** | %+.4e | %+.4e | %+.4e |\n",
                    force_data.Fz, force_data.Fz_pressure, force_data.Fz_viscous)
            println(io, "")

            println(io, "### Moments [N·m]")
            println(io, "")
            println(io, "| Component | Value |")
            println(io, "|-----------|-------|")
            @printf(io, "| **Mx (roll)** | %+.4e |\n", force_data.Mx)
            @printf(io, "| **My (pitch)** | %+.4e |\n", force_data.My)
            @printf(io, "| **Mz (yaw)** | %+.4e |\n", force_data.Mz)
            println(io, "")

            println(io, "### Coefficients (instantaneous)")
            println(io, "")
            println(io, "| Coefficient | Value |")
            println(io, "|-------------|-------|")
            @printf(io, "| **Cd** | %+.6f |\n", force_data.Cd)
            @printf(io, "| **Cdp (pressure drag)** | %+.6f |\n", Cdp_final)
            @printf(io, "| **Cdv (viscous drag)** | %+.6f |\n", Cdv_final)
            @printf(io, "| **Cl** | %+.6f |\n", force_data.Cl)
            @printf(io, "| **Cs** | %+.6f |\n", force_data.Cs)
            @printf(io, "| **Cmx** | %+.6f |\n", force_data.Cmx)
            @printf(io, "| **Cmy** | %+.6f |\n", force_data.Cmy)
            @printf(io, "| **Cmz** | %+.6f |\n", force_data.Cmz)
            println(io, "")
        end

        if force_accm !== nothing && force_accm.n_samples > 0
            println(io, "## Aerodynamic Forces — Time-Averaged")
            println(io, "")
            @printf(io, "*%d samples after ramp (step %d to %d)*\n\n", force_accm.n_samples, RAMP_STEPS, STEPS)

            n = force_accm.n_samples
            inv_n = 1.0 / n

            avg_Fx_p = force_accm.sum_Fx_p * inv_n
            avg_Fy_p = force_accm.sum_Fy_p * inv_n
            avg_Fz_p = force_accm.sum_Fz_p * inv_n
            avg_Fx_v = force_accm.sum_Fx_v * inv_n
            avg_Fy_v = force_accm.sum_Fy_v * inv_n
            avg_Fz_v = force_accm.sum_Fz_v * inv_n
            avg_Mx   = force_accm.sum_Mx * inv_n
            avg_My   = force_accm.sum_My * inv_n
            avg_Mz   = force_accm.sum_Mz * inv_n

            avg_Fx = avg_Fx_p + avg_Fx_v
            avg_Fy = avg_Fy_p + avg_Fy_v
            avg_Fz = avg_Fz_p + avg_Fz_v

            avg_Cd  = F_ref > 1e-10 ? avg_Fx / F_ref : 0.0
            avg_Cl  = F_ref > 1e-10 ? avg_Fz / F_ref : 0.0
            avg_Cs  = F_ref > 1e-10 ? avg_Fy / F_ref : 0.0
            avg_Cdp = F_ref > 1e-10 ? avg_Fx_p / F_ref : 0.0
            avg_Cdv = F_ref > 1e-10 ? avg_Fx_v / F_ref : 0.0
            avg_Cmx = M_ref > 1e-10 ? avg_Mx / M_ref : 0.0
            avg_Cmy = M_ref > 1e-10 ? avg_My / M_ref : 0.0
            avg_Cmz = M_ref > 1e-10 ? avg_Mz / M_ref : 0.0

            pct_pressure = abs(avg_Fx) > 1e-10 ? 100.0 * abs(avg_Fx_p) / (abs(avg_Fx_p) + abs(avg_Fx_v)) : 0.0

            println(io, "### Averaged Forces [N]")
            println(io, "")
            println(io, "| Component | Total | Pressure | Viscous |")
            println(io, "|-----------|-------|----------|---------|")
            @printf(io, "| **Fx (drag)** | %+.4e | %+.4e | %+.4e |\n", avg_Fx, avg_Fx_p, avg_Fx_v)
            @printf(io, "| **Fy (side)** | %+.4e | %+.4e | %+.4e |\n", avg_Fy, avg_Fy_p, avg_Fy_v)
            @printf(io, "| **Fz (lift)** | %+.4e | %+.4e | %+.4e |\n", avg_Fz, avg_Fz_p, avg_Fz_v)
            println(io, "")

            println(io, "### Averaged Moments [N·m]")
            println(io, "")
            println(io, "| Component | Value |")
            println(io, "|-----------|-------|")
            @printf(io, "| **Mx (roll)** | %+.4e |\n", avg_Mx)
            @printf(io, "| **My (pitch)** | %+.4e |\n", avg_My)
            @printf(io, "| **Mz (yaw)** | %+.4e |\n", avg_Mz)
            println(io, "")

            println(io, "### Averaged Coefficients")
            println(io, "")
            println(io, "| Coefficient | Value |")
            println(io, "|-------------|-------|")
            @printf(io, "| **Cd** | %+.6f |\n", avg_Cd)
            @printf(io, "| **Cdp (pressure drag)** | %+.6f |\n", avg_Cdp)
            @printf(io, "| **Cdv (viscous drag)** | %+.6f |\n", avg_Cdv)
            @printf(io, "| **Cl** | %+.6f |\n", avg_Cl)
            @printf(io, "| **Cs** | %+.6f |\n", avg_Cs)
            @printf(io, "| **Cmx** | %+.6f |\n", avg_Cmx)
            @printf(io, "| **Cmy** | %+.6f |\n", avg_Cmy)
            @printf(io, "| **Cmz** | %+.6f |\n", avg_Cmz)
            println(io, "")

            @printf(io, "**Drag breakdown:** %.1f%% pressure, %.1f%% viscous\n\n", pct_pressure, 100.0 - pct_pressure)

            println(io, "### Coefficient Ranges (instantaneous min/max)")
            println(io, "")
            println(io, "| Coefficient | Min | Max |")
            println(io, "|-------------|-----|-----|")
            @printf(io, "| **Cd** | %+.6f | %+.6f |\n", force_accm.min_Cd, force_accm.max_Cd)
            @printf(io, "| **Cl** | %+.6f | %+.6f |\n", force_accm.min_Cl, force_accm.max_Cl)
            @printf(io, "| **Cs** | %+.6f | %+.6f |\n", force_accm.min_Cs, force_accm.max_Cs)
            println(io, "")
        elseif force_data === nothing
            println(io, "## Aerodynamic Forces")
            println(io, "")
            println(io, "*Force computation was disabled for this run.*")
            println(io, "")
        end

        # ══════════════════════════════════════════════════════════════
        # PERFORMANCE
        # ══════════════════════════════════════════════════════════════
        println(io, "## Performance")
        println(io, "")
        println(io, "| Metric | Value |")
        println(io, "|--------|-------|")
        @printf(io, "| **Wall time** | %.1f s |\n", total_time)
        @printf(io, "| **Average MLUPS** | %.1f |\n", avg_mlups)
        @printf(io, "| **Total cells** | %.2f M |\n", total_cells / 1e6)
        @printf(io, "| **Total timesteps** | %d |\n", STEPS)
        @printf(io, "| **Cell updates** | %.2e |\n", Float64(total_cells) * STEPS)
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # OUTPUT FIELDS
        # ══════════════════════════════════════════════════════════════
        println(io, "## Output Fields")
        println(io, "")
        println(io, "| Field | Exported |")
        println(io, "|-------|----------|")
        @printf(io, "| Density | %s |\n", OUTPUT_DENSITY ? "Yes" : "No")
        @printf(io, "| Velocity | %s |\n", OUTPUT_VELOCITY ? "Yes" : "No")
        @printf(io, "| Velocity magnitude | %s |\n", OUTPUT_VEL_MAG ? "Yes" : "No")
        @printf(io, "| Vorticity | %s |\n", OUTPUT_VORTICITY ? "Yes" : "No")
        @printf(io, "| Obstacle | %s |\n", OUTPUT_OBSTACLE ? "Yes" : "No")
        @printf(io, "| Level | %s |\n", OUTPUT_LEVEL ? "Yes" : "No")
        @printf(io, "| Bouzidi | %s |\n", OUTPUT_BOUZIDI ? "Yes" : "No")
        @printf(io, "| Net force export | %s |\n", NET_FORCE_EXPORT_ENABLED ? "Yes" : "No")
        println(io, "")

        # ══════════════════════════════════════════════════════════════
        # FOOTER
        # ══════════════════════════════════════════════════════════════
        println(io, "---")
        println(io, "*Generated by LUDWIG v1.10 LBM Solver*")

    end  # close file

    println("[Summary] Analysis summary written to: $(summary_path)")
end


"""
    build_case_result(params, force_data, force_accm, total_cells, total_time, start_dt, end_dt)

Extracts the key results from a completed simulation into a lightweight NamedTuple
that can be collected across multiple cases for the batch summary.
"""
function build_case_result(params, force_data, force_accm,
                           total_cells::Int, total_time::Float64,
                           start_datetime::DateTime, end_datetime::DateTime)

    q_inf = 0.5 * params.rho_physical * params.u_physical^2
    F_ref = q_inf * params.reference_area
    M_ref = F_ref * params.reference_chord
    avg_mlups = total_time > 0 ? (total_cells * STEPS) / (total_time * 1e6) : 0.0

    # Final instantaneous coefficients
    Cd  = force_data !== nothing ? force_data.Cd  : 0.0
    Cl  = force_data !== nothing ? force_data.Cl  : 0.0
    Cs  = force_data !== nothing ? force_data.Cs  : 0.0
    Cdp = (force_data !== nothing && F_ref > 1e-10) ? force_data.Fx_pressure / F_ref : 0.0
    Cdv = (force_data !== nothing && F_ref > 1e-10) ? force_data.Fx_viscous  / F_ref : 0.0
    Cmx = force_data !== nothing ? force_data.Cmx : 0.0
    Cmy = force_data !== nothing ? force_data.Cmy : 0.0
    Cmz = force_data !== nothing ? force_data.Cmz : 0.0

    # Averaged coefficients
    n_samples = 0
    avg_Cd = 0.0; avg_Cl = 0.0; avg_Cs = 0.0
    avg_Cdp = 0.0; avg_Cdv = 0.0
    avg_Cmx = 0.0; avg_Cmy = 0.0; avg_Cmz = 0.0
    _min_Cd = 0.0; _max_Cd = 0.0
    _min_Cl = 0.0; _max_Cl = 0.0
    _min_Cs = 0.0; _max_Cs = 0.0

    if force_accm !== nothing && force_accm.n_samples > 0
        n = force_accm.n_samples
        inv_n = 1.0 / n
        n_samples = n

        avg_Fx_p = force_accm.sum_Fx_p * inv_n
        avg_Fx_v = force_accm.sum_Fx_v * inv_n
        avg_Fy_p = force_accm.sum_Fy_p * inv_n
        avg_Fz_p = force_accm.sum_Fz_p * inv_n
        avg_Fz_v = force_accm.sum_Fz_v * inv_n
        avg_Fx = avg_Fx_p + avg_Fx_v
        avg_Fy = avg_Fy_p + force_accm.sum_Fy_v * inv_n
        avg_Fz = avg_Fz_p + avg_Fz_v

        avg_Cd  = F_ref > 1e-10 ? avg_Fx / F_ref : 0.0
        avg_Cl  = F_ref > 1e-10 ? avg_Fz / F_ref : 0.0
        avg_Cs  = F_ref > 1e-10 ? avg_Fy / F_ref : 0.0
        avg_Cdp = F_ref > 1e-10 ? avg_Fx_p / F_ref : 0.0
        avg_Cdv = F_ref > 1e-10 ? avg_Fx_v / F_ref : 0.0

        avg_Mx = force_accm.sum_Mx * inv_n
        avg_My = force_accm.sum_My * inv_n
        avg_Mz = force_accm.sum_Mz * inv_n
        avg_Cmx = M_ref > 1e-10 ? avg_Mx / M_ref : 0.0
        avg_Cmy = M_ref > 1e-10 ? avg_My / M_ref : 0.0
        avg_Cmz = M_ref > 1e-10 ? avg_Mz / M_ref : 0.0

        _min_Cd = force_accm.min_Cd; _max_Cd = force_accm.max_Cd
        _min_Cl = force_accm.min_Cl; _max_Cl = force_accm.max_Cl
        _min_Cs = force_accm.min_Cs; _max_Cs = force_accm.max_Cs
    end

    return (
        case_name  = basename(CASE_DIR),
        status     = :completed,
        error_msg  = "",
        re_number  = params.re_number,
        u_physical = params.u_physical,
        alpha_deg  = Float64(ALPHA_DEG),
        beta_deg   = Float64(BETA_DEG),
        total_cells = total_cells,
        wall_time  = total_time,
        mlups      = avg_mlups,
        steps      = STEPS,
        start_time = start_datetime,
        end_time   = end_datetime,
        # Final instantaneous
        Cd = Cd, Cl = Cl, Cs = Cs,
        Cdp = Cdp, Cdv = Cdv,
        Cmx = Cmx, Cmy = Cmy, Cmz = Cmz,
        # Averaged
        avg_Cd = avg_Cd, avg_Cl = avg_Cl, avg_Cs = avg_Cs,
        avg_Cdp = avg_Cdp, avg_Cdv = avg_Cdv,
        avg_Cmx = avg_Cmx, avg_Cmy = avg_Cmy, avg_Cmz = avg_Cmz,
        n_samples = n_samples,
        min_Cd = _min_Cd, max_Cd = _max_Cd,
        min_Cl = _min_Cl, max_Cl = _max_Cl,
        min_Cs = _min_Cs, max_Cs = _max_Cs,
        collision_operator = COLLISION_OPERATOR,
        ma_physical = params.u_physical / 343.0,
        compressibility_correction = COMPRESSIBILITY_CORRECTION,
        output_dir = abspath(OUT_DIR)
    )
end


"""
    write_cases_summary(output_dir, case_results, batch_start, batch_end)

Writes a multi-case summary markdown file next to cases_to_run.yaml,
with the filename `cases_summary_YYYYMMDD_HHMMSS.md`.
"""
function write_cases_summary(output_dir::String, case_results::Vector,
                              batch_start::DateTime, batch_end::DateTime)

    timestamp_str = Dates.format(batch_start, "yyyymmdd_HHMMSS")
    filename = "cases_summary_$(timestamp_str).md"
    filepath = joinpath(output_dir, filename)

    total_wall = sum(r.wall_time for r in case_results)
    n_completed = count(r -> r.status == :completed, case_results)
    n_failed    = count(r -> r.status == :failed, case_results)

    open(filepath, "w") do io

        # ── Header ──
        println(io, "# Multi-Case Summary — LUDWIG v1.10")
        println(io, "")
        println(io, "## Batch Information")
        println(io, "")
        println(io, "| Field | Value |")
        println(io, "|-------|-------|")
        @printf(io, "| **Start time** | %s |\n", Dates.format(batch_start, "yyyy-mm-dd HH:MM:SS"))
        @printf(io, "| **End time** | %s |\n", Dates.format(batch_end, "yyyy-mm-dd HH:MM:SS"))
        batch_elapsed = Dates.value(batch_end - batch_start) / 1000.0  # seconds
        @printf(io, "| **Total elapsed** | %.1f s (%.2f min) |\n", batch_elapsed, batch_elapsed / 60.0)
        @printf(io, "| **Total solver time** | %.1f s (%.2f min) |\n", total_wall, total_wall / 60.0)
        @printf(io, "| **Cases** | %d total (%d completed, %d failed) |\n",
                length(case_results), n_completed, n_failed)
        println(io, "")

        # ── Compact comparison table: time-averaged coefficients ──
        println(io, "## Aerodynamic Coefficients — Time-Averaged")
        println(io, "")
        println(io, "| Case | Re | Ma | U [m/s] | α [deg] | β [deg] | Cd | Cdp | Cdv | Cl | Cs | Cmy | Status |")
        println(io, "|------|---:|---:|--------:|--------:|--------:|---:|----:|----:|---:|---:|----:|--------|")
        for r in case_results
            if r.status == :completed && r.n_samples > 0
                @printf(io, "| %s | %.2e | %.3f | %.2f | %.1f | %.1f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %s |\n",
                        r.case_name, r.re_number, r.ma_physical, r.u_physical,
                        r.alpha_deg, r.beta_deg,
                        r.avg_Cd, r.avg_Cdp, r.avg_Cdv,
                        r.avg_Cl, r.avg_Cs, r.avg_Cmy,
                        "Averaged ($(r.n_samples) samples)")
            elseif r.status == :completed
                @printf(io, "| %s | %.2e | %.3f | %.2f | %.1f | %.1f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %s |\n",
                        r.case_name, r.re_number, r.ma_physical, r.u_physical,
                        r.alpha_deg, r.beta_deg,
                        r.Cd, r.Cdp, r.Cdv,
                        r.Cl, r.Cs, r.Cmy,
                        "Final instant.")
            else
                @printf(io, "| %s | — | — | — | — | — | — | — | — | — | — | — | FAILED |\n", r.case_name)
            end
        end
        println(io, "")

        # ── Coefficient ranges ──
        completed_with_avg = filter(r -> r.status == :completed && r.n_samples > 0, case_results)
        if !isempty(completed_with_avg)
            println(io, "## Coefficient Ranges (instantaneous min/max per case)")
            println(io, "")
            println(io, "| Case | Re | Cd min | Cd max | Cl min | Cl max | Cs min | Cs max |")
            println(io, "|------|---:|-------:|-------:|-------:|-------:|-------:|-------:|")
            for r in completed_with_avg
                @printf(io, "| %s | %.2e | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f |\n",
                        r.case_name, r.re_number,
                        r.min_Cd, r.max_Cd,
                        r.min_Cl, r.max_Cl,
                        r.min_Cs, r.max_Cs)
            end
            println(io, "")
        end

        # ── Performance summary ──
        println(io, "## Performance Summary")
        println(io, "")
        println(io, "| Case | Cells [M] | Steps | Wall time [s] | MLUPS | Operator |")
        println(io, "|------|----------:|------:|--------------:|------:|----------|")
        for r in case_results
            if r.status == :completed
                @printf(io, "| %s | %.2f | %d | %.1f | %.1f | %s |\n",
                        r.case_name, r.total_cells / 1e6, r.steps,
                        r.wall_time, r.mlups, r.collision_operator)
            else
                @printf(io, "| %s | — | — | — | — | FAILED |\n", r.case_name)
            end
        end
        println(io, "")

        # ── Per-case timing ──
        println(io, "## Per-Case Timing")
        println(io, "")
        println(io, "| Case | Start | End | Duration |")
        println(io, "|------|-------|-----|----------|")
        for r in case_results
            if r.status == :completed
                dur = r.wall_time
                @printf(io, "| %s | %s | %s | %.1f s (%.2f min) |\n",
                        r.case_name,
                        Dates.format(r.start_time, "HH:MM:SS"),
                        Dates.format(r.end_time, "HH:MM:SS"),
                        dur, dur / 60.0)
            else
                @printf(io, "| %s | — | — | FAILED |\n", r.case_name)
            end
        end
        println(io, "")

        # ── Failed cases ──
        if n_failed > 0
            println(io, "## Failed Cases")
            println(io, "")
            for r in case_results
                if r.status == :failed
                    @printf(io, "- **%s**: `%s`\n", r.case_name, r.error_msg)
                end
            end
            println(io, "")
        end

        # ── Output directories ──
        println(io, "## Output Directories")
        println(io, "")
        for r in case_results
            if r.status == :completed
                @printf(io, "- **%s**: `%s`\n", r.case_name, r.output_dir)
            end
        end
        println(io, "")

        # ── Footer ──
        println(io, "---")
        println(io, "*Generated by LUDWIG v1.10 LBM Solver*")
    end

    println("[CasesSummary] Multi-case summary written to: $(filepath)")
end
