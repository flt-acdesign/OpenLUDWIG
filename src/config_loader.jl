// # FILE: .\src\config_loader.jl
using YAML
using Printf

"""
CONFIG_LOADER.JL - Configuration Management

Handles loading of user settings from YAML files and defining
global control variables for the simulation.
"""

# --- Global Configuration Storage ---
global CFG = Dict()
global CASE_DIR = ""
global STL_FILENAME = ""
global STL_FILE = ""
global STL_SCALE = 1.0
global OUT_DIR_NAME = "RESULTS"
global OUT_DIR = ""
global SURFACE_RESOLUTION = 200
global NUM_LEVELS_CONFIG = 0

global SYMMETRIC_ANALYSIS = false
global REFERENCE_AREA_FULL_MODEL = 0.0
global REFERENCE_AREA_CONFIG = 0.0
global REFERENCE_CHORD_CONFIG = 0.0
global REFERENCE_LENGTH_FOR_MESHING = 0.0
global REFERENCE_DIMENSION = :x

global FLUID_DENSITY = 1.225
global FLUID_KINEMATIC_VISCOSITY = 1.5e-5
global FLOW_VELOCITY = 10.0

global STEPS = 1000
global RAMP_STEPS = 4000
global OUTPUT_FREQ = 100

global OUTPUT_DENSITY = true
global OUTPUT_VELOCITY = true
global OUTPUT_VEL_MAG = true
global OUTPUT_VORTICITY = true
global OUTPUT_OBSTACLE = true
global OUTPUT_LEVEL = true
global OUTPUT_BOUZIDI = true

global U_TARGET = 0.05f0
global C_SMAGO = 0.20f0
global TAU_MIN = 0.505f0
global TAU_SAFETY_FACTOR = 1.1f0
global INLET_TURBULENCE_INTENSITY = 0.01f0

# --- Stability Parameters ---
global NU_SGS_BACKGROUND = 0.0005f0         
global SPONGE_BLEND_DISTRIBUTIONS = true    
global TEMPORAL_INTERPOLATION = true        
global INTERFACE_FILTERING = false

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

global DIAG_FREQ = 100
global STABILITY_CHECK_ENABLED = true
global PRINT_TAU_WARNING = true

global GPU_ASYNC_DEPTH = 3
global USE_STREAMS = true
global PREFETCH_NEIGHBORS = true

# --- Helper Functions ---

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
    global CASE_DIR = abspath(joinpath(@__DIR__, "../CASES", case_folder_name))
    !isdir(CASE_DIR) && error("Case folder not found: $CASE_DIR")
    
    config_path = joinpath(CASE_DIR, "config.yaml")
    !isfile(config_path) && error("config.yaml not found: $config_path")
    
    println("[Init] Loading: $case_folder_name")
    global CFG = YAML.load_file(config_path)
    
    global STL_FILENAME = safe_get(CFG, "basic", "stl_file")
    global STL_FILE = joinpath(CASE_DIR, STL_FILENAME)
    global STL_SCALE = Float64(safe_get(CFG, "basic", "stl_scale"))
    global OUT_DIR_NAME = safe_get(CFG, "basic", "simulation", "output_dir")
    global OUT_DIR = joinpath(CASE_DIR, OUT_DIR_NAME)
    global SURFACE_RESOLUTION = Int(safe_get(CFG, "basic", "surface_resolution"))
    global NUM_LEVELS_CONFIG = Int(safe_get(CFG, "basic", "num_levels"))
    
    global SYMMETRIC_ANALYSIS = safe_get(CFG, "advanced", "refinement", "symmetric_analysis"; default=false)
    global REFERENCE_AREA_FULL_MODEL = Float64(safe_get(CFG, "basic", "reference_area_of_full_model"; default=0.0))
    global REFERENCE_AREA_CONFIG = SYMMETRIC_ANALYSIS ? REFERENCE_AREA_FULL_MODEL/2.0 : REFERENCE_AREA_FULL_MODEL
    global REFERENCE_CHORD_CONFIG = Float64(safe_get(CFG, "basic", "reference_chord"; default=0.0))
    global REFERENCE_LENGTH_FOR_MESHING = Float64(safe_get(CFG, "basic", "reference_length_for_meshing"; default=0.0))
    global REFERENCE_DIMENSION = Symbol(safe_get(CFG, "basic", "reference_dimension"; default="x"))
    
    global FLUID_DENSITY = Float64(safe_get(CFG, "basic", "fluid", "density"; default=1.225))
    global FLUID_KINEMATIC_VISCOSITY = Float64(safe_get(CFG, "basic", "fluid", "kinematic_viscosity"; default=1.5e-5))
    global FLOW_VELOCITY = Float64(safe_get(CFG, "basic", "flow", "velocity"; default=10.0))
    
    global STEPS = Int(safe_get(CFG, "basic", "simulation", "steps"))
    global RAMP_STEPS = Int(safe_get(CFG, "basic", "simulation", "ramp_steps"))
    global OUTPUT_FREQ = Int(safe_get(CFG, "basic", "simulation", "output_freq"))
    
    global OUTPUT_DENSITY = safe_get(CFG, "basic", "simulation", "output_fields", "density"; default=true)
    global OUTPUT_VELOCITY = safe_get(CFG, "basic", "simulation", "output_fields", "velocity"; default=true)
    global OUTPUT_VEL_MAG = safe_get(CFG, "basic", "simulation", "output_fields", "velocity_magnitude"; default=true)
    global OUTPUT_VORTICITY = safe_get(CFG, "basic", "simulation", "output_fields", "vorticity"; default=true)
    global OUTPUT_OBSTACLE = safe_get(CFG, "basic", "simulation", "output_fields", "obstacle"; default=true)
    global OUTPUT_LEVEL = safe_get(CFG, "basic", "simulation", "output_fields", "level"; default=true)
    global OUTPUT_BOUZIDI = safe_get(CFG, "basic", "simulation", "output_fields", "bouzidi"; default=true)
    
    global U_TARGET = Float32(safe_get(CFG, "advanced", "numerics", "u_lattice"; default=0.01))
    global C_SMAGO = Float32(safe_get(CFG, "advanced", "numerics", "c_wale"; default=0.20))
    global TAU_MIN = Float32(safe_get(CFG, "advanced", "numerics", "tau_min"; default=0.505))
    global TAU_SAFETY_FACTOR = Float32(safe_get(CFG, "advanced", "numerics", "tau_safety_factor"; default=1.0))
    global INLET_TURBULENCE_INTENSITY = Float32(safe_get(CFG, "advanced", "numerics", "inlet_turbulence_intensity"; default=0.01))
    
    global NU_SGS_BACKGROUND = Float32(safe_get(CFG, "advanced", "numerics", "nu_sgs_background"; default=0.0005))
    global SPONGE_BLEND_DISTRIBUTIONS = safe_get(CFG, "advanced", "numerics", "sponge_blend_distributions"; default=true)
    global TEMPORAL_INTERPOLATION = safe_get(CFG, "advanced", "numerics", "temporal_interpolation"; default=true)
    
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
    
    global DIAG_FREQ = Int(safe_get(CFG, "advanced", "diagnostics", "freq"; default=500))
    global STABILITY_CHECK_ENABLED = safe_get(CFG, "advanced", "diagnostics", "stability_check"; default=true)
    global PRINT_TAU_WARNING = safe_get(CFG, "advanced", "diagnostics", "print_tau_warning"; default=true)
    global FORCE_OUTPUT_FREQ = FORCE_OUTPUT_FREQ_CONFIG == 0 ? DIAG_FREQ : FORCE_OUTPUT_FREQ_CONFIG
    
    global GPU_ASYNC_DEPTH = Int(safe_get(CFG, "advanced", "gpu", "async_depth"; default=8))
    global USE_STREAMS = safe_get(CFG, "advanced", "gpu", "use_streams"; default=true)
    global PREFETCH_NEIGHBORS = safe_get(CFG, "advanced", "gpu", "prefetch_neighbors"; default=true)
    
    # Flag to reset domain parameters when a new case loads
    if isdefined(Main, :DOMAIN_PARAMS)
        DOMAIN_PARAMS.initialized = false
    end
    
    println("[Init] ═══════════════════════════════════════════════════════")
    println("[Init] STABILITY SETTINGS:")
    @printf("[Init]    Background ν_sgs: %.6f → τ_eff_min ≈ %.4f\n", NU_SGS_BACKGROUND, 0.5 + 3*NU_SGS_BACKGROUND)
    println("[Init]    Sponge f-blending: $SPONGE_BLEND_DISTRIBUTIONS")
    println("[Init]    Temporal interpolation: $TEMPORAL_INTERPOLATION")
    println("[Init] ═══════════════════════════════════════════════════════")
end