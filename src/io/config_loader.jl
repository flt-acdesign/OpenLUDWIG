using YAML
using Printf


global CFG = Dict()
global CASE_DIR = ""
global STL_FILENAME = ""
global STL_FILE = ""
global STL_FILENAMES = String[]    # All STL filenames (always a vector)
global STL_FILES = String[]        # Full paths to all STL files
global STL_SCALE = 1.0
global OUT_DIR_NAME = "RESULTS"
global OUT_DIR = ""
global SURFACE_RESOLUTION = 200
global NUM_LEVELS_CONFIG = 0
global MINIMUM_FACET_SIZE = 0.0

global SYMMETRIC_ANALYSIS = false
global ALPHA_DEG = 0.0
global BETA_DEG = 0.0
global REFERENCE_AREA_FULL_MODEL = 0.0
global REFERENCE_AREA_CONFIG = 0.0
global REFERENCE_CHORD_CONFIG = 0.0
global REFERENCE_LENGTH_FOR_MESHING = 0.0
global REFERENCE_DIMENSION = :x

global FLUID_DENSITY = 1.225
global FLUID_KINEMATIC_VISCOSITY = 1.5e-5
global FLOW_VELOCITY = 10.0

global STEPS = 1000
global RAMP_STEPS = 1500
global OUTPUT_FREQ = 100
global DOMAIN_LENGTH_COVERED = 0.0  # 0.0 = use 'steps' directly; >0 = compute steps from domain traversals

global OUTPUT_DENSITY = true
global OUTPUT_VELOCITY = true
global OUTPUT_VEL_MAG = true
global OUTPUT_VORTICITY = true
global OUTPUT_OBSTACLE = true
global OUTPUT_LEVEL = true
global OUTPUT_BOUZIDI = true

global U_TARGET = 0.08f0
global C_SMAGO = 0.20f0
global TAU_MIN = 0.501f0
global TAU_SAFETY_FACTOR = 1.0f0
global INLET_TURBULENCE_INTENSITY = 0.01f0
global COLLISION_OPERATOR = :cumulant

global NU_SGS_BACKGROUND = 0.0f0
global SPONGE_BLEND_DISTRIBUTIONS = false
global TEMPORAL_INTERPOLATION = true
global INTERFACE_FILTERING = false

global CUMULANT_OMEGA_BULK      = 1.0f0
global CUMULANT_OMEGA_3         = 1.0f0
global CUMULANT_OMEGA_4         = 1.0f0
global CUMULANT_ADAPTIVE_OMEGA4 = true
global CUMULANT_LAMBDA_PARAM    = Float32(1.0/6.0)
global CUMULANT_LIMITER         = :factored

const VALID_CUMULANT_LIMITERS = Set([:none, :factored, :positivity])

# Compressibility correction (Phase 1): extends valid Ma range to ~0.6–0.9
# by including higher-order velocity terms in the equilibrium that are
# normally truncated at O(u²).  No extra memory or distributions required.
global COMPRESSIBILITY_CORRECTION = false

# Double Distribution Function (Phase 2): thermal/energy transport
# Adds 27 extra distributions (g) for temperature field.
# When disabled, solver behaves identically to standard isothermal LBM.
global DDF_ENABLED = false
global DDF_PRANDTL = 0.71
global DDF_T_INLET = 1.0
global DDF_T_WALL = 1.0
global DDF_T_INITIAL = 1.0
global DDF_WALL_BC = :adiabatic   # :adiabatic or :isothermal

global AUTO_LEVELS = false
global MAX_LEVELS = 12
global MIN_COARSE_BLOCKS = 4
global WALL_MODEL_ENABLED = false
global WALL_MODEL_TYPE = :equilibrium
global WALL_MODEL_YPLUS_TARGET = 30.0

global DOMAIN_UPSTREAM = 0.75
global DOMAIN_DOWNSTREAM = 1.5
global DOMAIN_LATERAL = 0.75
global DOMAIN_HEIGHT = 0.75
global SPONGE_THICKNESS = 0.10f0

global BLOCK_SIZE_CONFIG = 8
global REFINEMENT_MARGIN = 2
global REFINEMENT_STRATEGY = :geometry_first
global ENABLE_WAKE_REFINEMENT = false
global WAKE_REFINEMENT_LENGTH = 0.25
global WAKE_REFINEMENT_WIDTH_FACTOR = 0.1
global WAKE_REFINEMENT_HEIGHT_FACTOR = 0.1

global BOUNDARY_METHOD = :bounce_back
global BOUZIDI_LEVELS = 1
global Q_MIN_THRESHOLD = 0.001f0

global FORCE_COMPUTATION_ENABLED = false
global FORCE_OUTPUT_FREQ_CONFIG = 0
global FORCE_OUTPUT_FREQ = 0
global MOMENT_CENTER_CONFIG = [0.0, 0.0, 0.0]
global NET_FORCE_EXPORT_ENABLED = false

global DIAG_FREQ = 100
global STABILITY_CHECK_ENABLED = true
global PRINT_TAU_WARNING = true

global GPU_ASYNC_DEPTH = 8
global USE_STREAMS = true
global PREFETCH_NEIGHBORS = true

const VALID_COLLISION_OPERATORS = Set([:regularized_bgk, :cumulant])


function safe_get(dict, keys...; default=nothing)
    current = dict
    for (i, key) in enumerate(keys)
        if current === nothing || !isa(current, Dict) || !haskey(current, key)
            return default !== nothing ? default : error("Missing config key: $(join(keys[1:i], " → "))")
        end
        current = current[key]
    end
    return current === nothing && default !== nothing ? default : current
end

function load_case_configuration(case_folder_name::String)
    global CASE_DIR = abspath(joinpath(@__DIR__, "../../CASES", case_folder_name))
    !isdir(CASE_DIR) && error("Case folder not found: $CASE_DIR")
    
    config_path = joinpath(CASE_DIR, "config.yaml")
    !isfile(config_path) && error("config.yaml not found: $config_path")
    
    println("[Init] Loading: $case_folder_name")
    global CFG = YAML.load_file(config_path)
    
    local raw_stl = safe_get(CFG, "basic", "stl_file")
    if isa(raw_stl, String)
        global STL_FILENAMES = [raw_stl]
    elseif isa(raw_stl, Vector)
        global STL_FILENAMES = String[string(s) for s in raw_stl]
    else
        error("stl_file must be a string or list of strings, got: $(typeof(raw_stl))")
    end
    global STL_FILES = [joinpath(CASE_DIR, fn) for fn in STL_FILENAMES]
    # Backward-compatible aliases (first entry)
    global STL_FILENAME = STL_FILENAMES[1]
    global STL_FILE = STL_FILES[1]
    global STL_SCALE = Float64(safe_get(CFG, "basic", "stl_scale"))
    global OUT_DIR_NAME = safe_get(CFG, "basic", "simulation", "output_dir")
    global OUT_DIR = joinpath(CASE_DIR, OUT_DIR_NAME)
    global SURFACE_RESOLUTION = Int(safe_get(CFG, "basic", "surface_resolution"))
    global NUM_LEVELS_CONFIG = Int(safe_get(CFG, "basic", "num_levels"))
    global MINIMUM_FACET_SIZE = Float64(safe_get(CFG, "basic", "minimum_facet_size"; default=0.0))
    
    global SYMMETRIC_ANALYSIS = safe_get(CFG, "advanced", "refinement", "symmetric_analysis"; default=false)
    global ALPHA_DEG = Float64(safe_get(CFG, "basic", "alpha"; default=0.0))
    global BETA_DEG  = Float64(safe_get(CFG, "basic", "beta"; default=0.0))
    if SYMMETRIC_ANALYSIS && BETA_DEG != 0.0
        println("[Init] ⚠ beta=$(BETA_DEG)° ignored because symmetric_analysis=true (forcing beta=0)")
        global BETA_DEG = 0.0
    end
    global REFERENCE_AREA_FULL_MODEL = Float64(safe_get(CFG, "basic", "reference_area_of_full_model"; default=0.0))
    global REFERENCE_AREA_CONFIG = SYMMETRIC_ANALYSIS ? REFERENCE_AREA_FULL_MODEL/2.0 : REFERENCE_AREA_FULL_MODEL
    global REFERENCE_CHORD_CONFIG = Float64(safe_get(CFG, "basic", "reference_chord"; default=0.0))
    global REFERENCE_LENGTH_FOR_MESHING = Float64(safe_get(CFG, "basic", "reference_length_for_meshing"; default=0.0))
    global REFERENCE_DIMENSION = Symbol(safe_get(CFG, "basic", "reference_dimension"; default="x"))
    
    global FLUID_DENSITY = Float64(safe_get(CFG, "basic", "fluid", "density"; default=1.225))
    global FLUID_KINEMATIC_VISCOSITY = Float64(safe_get(CFG, "basic", "fluid", "kinematic_viscosity"; default=1.5e-5))
    global FLOW_VELOCITY = Float64(safe_get(CFG, "basic", "flow", "velocity"; default=10.0))
    
    global DOMAIN_LENGTH_COVERED = Float64(safe_get(CFG, "basic", "simulation", "domain_length_covered"; default=0.0))
    global STEPS = Int(safe_get(CFG, "basic", "simulation", "steps"; default=0))
    global RAMP_STEPS = Int(safe_get(CFG, "basic", "simulation", "ramp_steps"))
    global OUTPUT_FREQ = Int(safe_get(CFG, "basic", "simulation", "output_freq"))
    
    global OUTPUT_DENSITY = safe_get(CFG, "basic", "simulation", "output_fields", "density"; default=true)
    global OUTPUT_VELOCITY = safe_get(CFG, "basic", "simulation", "output_fields", "velocity"; default=true)
    global OUTPUT_VEL_MAG = safe_get(CFG, "basic", "simulation", "output_fields", "velocity_magnitude"; default=true)
    global OUTPUT_VORTICITY = safe_get(CFG, "basic", "simulation", "output_fields", "vorticity"; default=true)
    global OUTPUT_OBSTACLE = safe_get(CFG, "basic", "simulation", "output_fields", "obstacle"; default=true)
    global OUTPUT_LEVEL = safe_get(CFG, "basic", "simulation", "output_fields", "level"; default=true)
    global OUTPUT_BOUZIDI = safe_get(CFG, "basic", "simulation", "output_fields", "bouzidi"; default=true)
    
    global COLLISION_OPERATOR = Symbol(safe_get(CFG, "advanced", "numerics", "collision_operator"; default="cumulant"))
    if !(COLLISION_OPERATOR in VALID_COLLISION_OPERATORS)
        error("Unknown collision_operator: \"$COLLISION_OPERATOR\". Valid options: $(join(sort(collect(VALID_COLLISION_OPERATORS)), ", "))")
    end

    global U_TARGET = Float32(safe_get(CFG, "advanced", "numerics", "u_lattice"; default=0.08))
    global C_SMAGO = Float32(safe_get(CFG, "advanced", "numerics", "c_wale"; default=0.20))
    global TAU_MIN = Float32(safe_get(CFG, "advanced", "numerics", "tau_min"; default=0.501))
    global TAU_SAFETY_FACTOR = Float32(safe_get(CFG, "advanced", "numerics", "tau_safety_factor"; default=1.0))
    global INLET_TURBULENCE_INTENSITY = Float32(safe_get(CFG, "advanced", "numerics", "inlet_turbulence_intensity"; default=0.01))

    global NU_SGS_BACKGROUND = Float32(safe_get(CFG, "advanced", "numerics", "nu_sgs_background"; default=0.0))
    global SPONGE_BLEND_DISTRIBUTIONS = safe_get(CFG, "advanced", "numerics", "sponge_blend_distributions"; default=false)
    global TEMPORAL_INTERPOLATION = safe_get(CFG, "advanced", "numerics", "temporal_interpolation"; default=true)

    global CUMULANT_OMEGA_BULK      = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "omega_bulk"; default=1.0))
    global CUMULANT_OMEGA_3         = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "omega_3"; default=1.0))
    global CUMULANT_OMEGA_4         = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "omega_4"; default=1.0))
    global CUMULANT_ADAPTIVE_OMEGA4 = safe_get(CFG, "advanced", "numerics", "cumulant", "adaptive_omega_4"; default=true)
    global CUMULANT_LAMBDA_PARAM    = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "lambda_param"; default=1.0/6.0))

    local limiter_str = safe_get(CFG, "advanced", "numerics", "cumulant", "limiter"; default="factored")
    global CUMULANT_LIMITER = Symbol(limiter_str)
    if !(CUMULANT_LIMITER in VALID_CUMULANT_LIMITERS)
        error("Unknown cumulant limiter: \"$CUMULANT_LIMITER\". Valid: $(join(sort(collect(VALID_CUMULANT_LIMITERS)), ", "))")
    end

    global COMPRESSIBILITY_CORRECTION = safe_get(CFG, "advanced", "numerics", "compressibility_correction"; default=false)

    # Double Distribution Function (thermal field)
    # DDF is automatically enabled when compressibility_correction is true.
    # This avoids a separate switch — one flag controls both Phase 1 (O(u⁴)
    # equilibrium) and Phase 2 (thermal DDF), saving VRAM when disabled.
    global DDF_ENABLED = COMPRESSIBILITY_CORRECTION
    if DDF_ENABLED
        global DDF_PRANDTL = Float64(safe_get(CFG, "advanced", "thermal", "prandtl"; default=0.71))
        global DDF_T_INLET = Float64(safe_get(CFG, "advanced", "thermal", "t_inlet"; default=1.0))
        global DDF_T_WALL = Float64(safe_get(CFG, "advanced", "thermal", "t_wall"; default=1.0))
        global DDF_T_INITIAL = Float64(safe_get(CFG, "advanced", "thermal", "t_initial"; default=1.0))
        local ddf_wall_bc_str = safe_get(CFG, "advanced", "thermal", "wall_bc"; default="adiabatic")
        global DDF_WALL_BC = Symbol(ddf_wall_bc_str)
        if !(DDF_WALL_BC in Set([:adiabatic, :isothermal]))
            error("Unknown thermal wall_bc: \"$DDF_WALL_BC\". Valid: adiabatic, isothermal")
        end
    else
        global DDF_PRANDTL = 0.71
        global DDF_T_INLET = 1.0
        global DDF_T_WALL = 1.0
        global DDF_T_INITIAL = 1.0
        global DDF_WALL_BC = :adiabatic
    end

    global AUTO_LEVELS = safe_get(CFG, "advanced", "high_re", "auto_levels"; default=false)
    global MAX_LEVELS = Int(safe_get(CFG, "advanced", "high_re", "max_levels"; default=12))
    global MIN_COARSE_BLOCKS = Int(safe_get(CFG, "advanced", "high_re", "min_coarse_blocks"; default=4))
    global WALL_MODEL_ENABLED = safe_get(CFG, "advanced", "high_re", "wall_model", "enabled"; default=false)
    global WALL_MODEL_TYPE = Symbol(safe_get(CFG, "advanced", "high_re", "wall_model", "type"; default="equilibrium"))
    global WALL_MODEL_YPLUS_TARGET = Float64(safe_get(CFG, "advanced", "high_re", "wall_model", "y_plus_target"; default=30.0))
    
    global DOMAIN_UPSTREAM = Float64(safe_get(CFG, "advanced", "domain", "upstream"; default=0.75))
    global DOMAIN_DOWNSTREAM = Float64(safe_get(CFG, "advanced", "domain", "downstream"; default=1.5))
    global DOMAIN_LATERAL = Float64(safe_get(CFG, "advanced", "domain", "lateral"; default=0.75))
    global DOMAIN_HEIGHT = Float64(safe_get(CFG, "advanced", "domain", "height"; default=0.75))
    global SPONGE_THICKNESS = Float32(safe_get(CFG, "advanced", "domain", "sponge_thickness"; default=0.10))
    
    global BLOCK_SIZE_CONFIG = Int(safe_get(CFG, "advanced", "refinement", "block_size"; default=8))
    global REFINEMENT_MARGIN = Int(safe_get(CFG, "advanced", "refinement", "margin"; default=2))
    global REFINEMENT_STRATEGY = Symbol(safe_get(CFG, "advanced", "refinement", "strategy"; default="geometry_first"))
    global ENABLE_WAKE_REFINEMENT = safe_get(CFG, "advanced", "refinement", "wake_enabled"; default=false)
    global WAKE_REFINEMENT_LENGTH = Float64(safe_get(CFG, "advanced", "refinement", "wake_length"; default=0.25))
    global WAKE_REFINEMENT_WIDTH_FACTOR = Float64(safe_get(CFG, "advanced", "refinement", "wake_width_factor"; default=0.1))
    global WAKE_REFINEMENT_HEIGHT_FACTOR = Float64(safe_get(CFG, "advanced", "refinement", "wake_height_factor"; default=0.1))
    
    global BOUNDARY_METHOD = Symbol(safe_get(CFG, "advanced", "boundary", "method"; default="bouzidi"))
    global BOUZIDI_LEVELS = Int(safe_get(CFG, "advanced", "boundary", "bouzidi_levels"; default=1))
    global Q_MIN_THRESHOLD = Float32(safe_get(CFG, "advanced", "boundary", "q_min_threshold"; default=0.001))
    
    global FORCE_COMPUTATION_ENABLED = safe_get(CFG, "advanced", "forces", "enabled"; default=true)
    global FORCE_OUTPUT_FREQ_CONFIG = Int(safe_get(CFG, "advanced", "forces", "output_freq"; default=0))
    global MOMENT_CENTER_CONFIG = safe_get(CFG, "advanced", "forces", "moment_center"; default=[0.25, 0.0, 0.0])
    global NET_FORCE_EXPORT_ENABLED = safe_get(CFG, "advanced", "forces", "net_force_export"; default=false)
    
    global DIAG_FREQ = Int(safe_get(CFG, "advanced", "diagnostics", "freq"; default=500))
    global STABILITY_CHECK_ENABLED = safe_get(CFG, "advanced", "diagnostics", "stability_check"; default=true)
    global PRINT_TAU_WARNING = safe_get(CFG, "advanced", "diagnostics", "print_tau_warning"; default=true)
    global FORCE_OUTPUT_FREQ = FORCE_OUTPUT_FREQ_CONFIG == 0 ? DIAG_FREQ : FORCE_OUTPUT_FREQ_CONFIG
    
    global GPU_ASYNC_DEPTH = Int(safe_get(CFG, "advanced", "gpu", "async_depth"; default=8))
    global USE_STREAMS = safe_get(CFG, "advanced", "gpu", "use_streams"; default=true)
    global PREFETCH_NEIGHBORS = safe_get(CFG, "advanced", "gpu", "prefetch_neighbors"; default=true)
    
    if isdefined(Main, :DOMAIN_PARAMS)
        DOMAIN_PARAMS.initialized = false
    end
    
    println("[Init] ═══════════════════════════════════════════════════════")
    println("[Init] SOLVER SETTINGS:")
    if ALPHA_DEG != 0.0 || BETA_DEG != 0.0
        @printf("[Init]    Model rotation: α=%.2f° (around Y), β=%.2f° (around Z)\n", ALPHA_DEG, BETA_DEG)
    end
    println("[Init]    Collision operator: $COLLISION_OPERATOR")
    @printf("[Init]    Background ν_sgs: %.6f → τ_eff_min ≈ %.4f\n", NU_SGS_BACKGROUND, 0.5 + 3*NU_SGS_BACKGROUND)
    println("[Init]    Sponge f-blending: $SPONGE_BLEND_DISTRIBUTIONS")
    println("[Init]    Temporal interpolation: $TEMPORAL_INTERPOLATION")
    if length(STL_FILENAMES) > 1
        println("[Init]    STL files: $(length(STL_FILENAMES)) objects")
        for (si, fn) in enumerate(STL_FILENAMES)
            println("[Init]       [$si] $fn")
        end
    end
    if MINIMUM_FACET_SIZE > 0.0
        @printf("[Init]    Minimum facet size: %.6f m (subdivision enabled)\n", MINIMUM_FACET_SIZE)
    else
        println("[Init]    Minimum facet size: disabled (no subdivision)")
    end
    if COLLISION_OPERATOR == :cumulant
        println("[Init]    Cumulant settings:")
        @printf("[Init]       ω_bulk=%.3f, ω₃=%.3f, ω₄=%.3f\n", CUMULANT_OMEGA_BULK, CUMULANT_OMEGA_3, CUMULANT_OMEGA_4)
        println("[Init]       Adaptive ω₄: $CUMULANT_ADAPTIVE_OMEGA4" * (CUMULANT_ADAPTIVE_OMEGA4 ? @sprintf(" (Λ=%.4f)", CUMULANT_LAMBDA_PARAM) : ""))
        println("[Init]       Limiter: $CUMULANT_LIMITER")
    end
    println("[Init]    Compressibility correction: $COMPRESSIBILITY_CORRECTION")
    local _Ma_lattice = Float64(U_TARGET) * sqrt(3.0)
    local _Ma_physical = FLOW_VELOCITY / 343.0
    @printf("[Init]    Mach number: Ma_physical = %.4f, Ma_lattice = %.4f\n", _Ma_physical, _Ma_lattice)
    if _Ma_lattice > 0.3 && !COMPRESSIBILITY_CORRECTION
        println("[Init]    ⚠ WARNING: Ma_lattice > 0.3 without compressibility correction!")
    end
    if DDF_ENABLED
        println("[Init]       Thermal DDF: ENABLED (auto)")
        @printf("[Init]          Prandtl = %.3f\n", DDF_PRANDTL)
        @printf("[Init]          T_inlet = %.3f, T_wall = %.3f, T_initial = %.3f\n", DDF_T_INLET, DDF_T_WALL, DDF_T_INITIAL)
        println("[Init]          Wall BC: $DDF_WALL_BC")
    end
    if NET_FORCE_EXPORT_ENABLED
        println("[Init]    Net force export: ENABLED (two-sided probing)")
    end
    println("[Init] ═══════════════════════════════════════════════════════")
end