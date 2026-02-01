// # FILE: .\src\physics_scaling.jl
using Printf

"""
PHYSICS_SCALING.JL - Physical to Lattice Unit Conversion

Contains logic to compute Reynolds numbers, relaxation times,
and grid resolution requirements.
"""

const CS2 = 1.0f0 / 3.0f0
const CS4 = CS2 * CS2

mutable struct DomainParameters
    initialized::Bool
    num_levels::Int
    mesh_min::Tuple{Float64, Float64, Float64}
    mesh_max::Tuple{Float64, Float64, Float64}
    mesh_center::Tuple{Float64, Float64, Float64}
    mesh_extent::Tuple{Float64, Float64, Float64}
    reference_length::Float64
    reference_chord::Float64
    reference_area::Float64
    moment_center::Tuple{Float64, Float64, Float64}
    domain_min::Tuple{Float64, Float64, Float64}
    domain_max::Tuple{Float64, Float64, Float64}
    domain_size::Tuple{Float64, Float64, Float64}
    mesh_offset::Tuple{Float64, Float64, Float64}
    dx_fine::Float64
    dx_coarse::Float64
    dx_levels::Vector{Float64}
    nx_coarse::Int
    ny_coarse::Int
    nz_coarse::Int
    bx_max::Int
    by_max::Int
    bz_max::Int
    l_char::Float64
    nu_lattice::Float64
    tau_coarse::Float32
    tau_levels::Vector{Float32}
    cs2::Float32
    cs4::Float32
    re_number::Float64
    u_physical::Float64
    rho_physical::Float64
    nu_physical::Float64
    length_scale::Float64
    time_scale::Float64
    velocity_scale::Float64
    force_scale::Float64
    tau_fine::Float64
    tau_margin_percent::Float64
    wall_model_active::Bool
    y_plus_first_cell::Float64
    estimated_memory_gb::Float64
end

global DOMAIN_PARAMS = DomainParameters(
    false, 0, (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0),
    0.0, 0.0, 0.0, (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0),
    0.0, 0.0, Float64[], 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0f0, Float32[], 1f0/3f0, 1f0/9f0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0.0
)

function compute_tau_for_levels(Re::Float64, ref_length::Float64, resolution::Int, n_levels::Int, u_lattice::Float32)
    nu_lattice_fine = Float64(u_lattice) * resolution / Re
    return 3.0 * nu_lattice_fine + 0.5
end

function compute_max_levels_for_domain(domain_size::Float64, dx_fine::Float64, block_size::Int, min_blocks::Int)
    ratio = domain_size / (dx_fine * min_blocks * block_size)
    return ratio < 1.0 ? 1 : Int(floor(1 + log2(ratio)))
end

function print_re_analysis(Re::Float64, ref_length::Float64, resolution::Int, u_lattice::Float32)
    tau = compute_tau_for_levels(Re, ref_length, resolution, 1, u_lattice)
    println("\n[High-Re] ═══════════════════════════════════════════════════════")
    @printf("[High-Re] Re = %.2e, L = %.4f m, N = %d cells/L, U_lat = %.3f\n", Re, ref_length, resolution, u_lattice)
    @printf("[High-Re] Computed τ_fine = %.6f\n", tau)
    @printf("[High-Re] Effective τ_min (with ν_sgs_bg) = %.6f\n", max(tau, 0.5 + 3*Float64(NU_SGS_BACKGROUND)))
    tau < 0.501 && println("[High-Re] ⚠ τ ≈ 0.5 (stability limit) - background ν_sgs active")
    println("[High-Re] ═══════════════════════════════════════════════════════\n")
end

function compute_domain_from_mesh(mesh_min::Tuple{Float64,Float64,Float64}, mesh_max::Tuple{Float64,Float64,Float64})
    mesh_center = ((mesh_min[1]+mesh_max[1])/2, (mesh_min[2]+mesh_max[2])/2, (mesh_min[3]+mesh_max[3])/2)
    mesh_extent = (mesh_max[1]-mesh_min[1], mesh_max[2]-mesh_min[2], mesh_max[3]-mesh_min[3])
    
    ref_length = REFERENCE_LENGTH_FOR_MESHING > 0 ? REFERENCE_LENGTH_FOR_MESHING :
                 (REFERENCE_DIMENSION == :x ? mesh_extent[1] : REFERENCE_DIMENSION == :y ? mesh_extent[2] :
                  REFERENCE_DIMENSION == :z ? mesh_extent[3] : maximum(mesh_extent))
    
    ref_chord = REFERENCE_CHORD_CONFIG > 0 ? REFERENCE_CHORD_CONFIG : mesh_extent[1]
    ref_area = REFERENCE_AREA_CONFIG > 0 ? REFERENCE_AREA_CONFIG : 
               (SYMMETRIC_ANALYSIS ? mesh_extent[2]*mesh_extent[3]*2 : mesh_extent[2]*mesh_extent[3])
    
    moment_center_rel = (Float64(MOMENT_CENTER_CONFIG[1]), Float64(MOMENT_CENTER_CONFIG[2]), Float64(MOMENT_CENTER_CONFIG[3]))
    
    u_phys, nu_phys, rho_phys = FLOW_VELOCITY, FLUID_KINEMATIC_VISCOSITY, FLUID_DENSITY
    re_number = u_phys * ref_length / nu_phys
    
    tau_fine_computed = compute_tau_for_levels(re_number, ref_length, SURFACE_RESOLUTION, 1, U_TARGET)
    tau_fine = max(tau_fine_computed, TAU_MIN)
    
    print_re_analysis(re_number, ref_length, SURFACE_RESOLUTION, U_TARGET)
    
    domain_x = ref_length * (DOMAIN_UPSTREAM + DOMAIN_DOWNSTREAM) + mesh_extent[1]
    domain_y = SYMMETRIC_ANALYSIS ? (mesh_max[2] + ref_length*DOMAIN_LATERAL) : (mesh_extent[2] + 2*ref_length*DOMAIN_LATERAL)
    domain_z = mesh_extent[3] + 2*ref_length*DOMAIN_HEIGHT
    
    dx_fine = ref_length / SURFACE_RESOLUTION
    min_domain = min(domain_x, domain_y, domain_z)
    max_levels_domain = compute_max_levels_for_domain(min_domain, dx_fine, BLOCK_SIZE_CONFIG, MIN_COARSE_BLOCKS)
    
    num_levels = NUM_LEVELS_CONFIG > 0 ? min(NUM_LEVELS_CONFIG, max_levels_domain) : 
                 (AUTO_LEVELS ? min(max_levels_domain, MAX_LEVELS) : min(8, max_levels_domain))
    
    dx_coarse = dx_fine * 2^(num_levels - 1)
    dx_levels = [dx_fine * 2^(num_levels - lvl) for lvl in 1:num_levels]
    
    nx_coarse = max(BLOCK_SIZE_CONFIG, Int(ceil(ceil(domain_x/dx_coarse)/BLOCK_SIZE_CONFIG)*BLOCK_SIZE_CONFIG))
    ny_coarse = max(BLOCK_SIZE_CONFIG, Int(ceil(ceil(domain_y/dx_coarse)/BLOCK_SIZE_CONFIG)*BLOCK_SIZE_CONFIG))
    nz_coarse = max(BLOCK_SIZE_CONFIG, Int(ceil(ceil(domain_z/dx_coarse)/BLOCK_SIZE_CONFIG)*BLOCK_SIZE_CONFIG))
    
    domain_x, domain_y, domain_z = nx_coarse*dx_coarse, ny_coarse*dx_coarse, nz_coarse*dx_coarse
    bx_max, by_max, bz_max = nx_coarse÷BLOCK_SIZE_CONFIG, ny_coarse÷BLOCK_SIZE_CONFIG, nz_coarse÷BLOCK_SIZE_CONFIG
    
    mesh_x = ref_length * DOMAIN_UPSTREAM
    mesh_y = SYMMETRIC_ANALYSIS ? 0.0 : (domain_y/2 - mesh_center[2])
    mesh_z = domain_z/2 - mesh_center[3]
    mesh_offset = (mesh_x - mesh_min[1], mesh_y, mesh_z)
    
    length_scale = dx_fine
    velocity_scale = u_phys / Float64(U_TARGET)
    time_scale = length_scale / velocity_scale
    nu_lattice_fine = nu_phys * time_scale / (length_scale^2)
    
    tau_levels = Float32[]
    for lvl in 1:num_levels
        tau_lvl = lvl == num_levels ? tau_fine : 0.5 + (tau_fine - 0.5) * 2.0^(num_levels - lvl)
        push!(tau_levels, Float32(tau_lvl))
    end
    
    force_scale = rho_phys * length_scale^4 / time_scale^2
    moment_center_phys = (mesh_min[1] + mesh_offset[1] + moment_center_rel[1]*ref_chord,
                          mesh_center[2] + mesh_offset[2] + moment_center_rel[2]*ref_chord,
                          mesh_center[3] + mesh_offset[3] + moment_center_rel[3]*ref_chord)
    
    bytes_per_cell = TEMPORAL_INTERPOLATION ? 220 : 160
    total_cells_est = bx_max * by_max * bz_max * BLOCK_SIZE_CONFIG^3
    for lvl in 2:num_levels; total_cells_est += Int(ceil(total_cells_est * 0.08)); end
    estimated_memory_gb = total_cells_est * bytes_per_cell / 1e9
    
    p = DOMAIN_PARAMS
    p.initialized = true
    p.num_levels = num_levels
    p.mesh_min, p.mesh_max, p.mesh_center, p.mesh_extent = mesh_min, mesh_max, mesh_center, mesh_extent
    p.reference_length, p.reference_chord, p.reference_area = ref_length, ref_chord, ref_area
    p.moment_center = moment_center_phys
    p.domain_min, p.domain_max = (0.0, 0.0, 0.0), (domain_x, domain_y, domain_z)
    p.domain_size, p.mesh_offset = (domain_x, domain_y, domain_z), mesh_offset
    p.dx_fine, p.dx_coarse, p.dx_levels = dx_fine, dx_coarse, dx_levels
    p.nx_coarse, p.ny_coarse, p.nz_coarse = nx_coarse, ny_coarse, nz_coarse
    p.bx_max, p.by_max, p.bz_max = bx_max, by_max, bz_max
    p.l_char, p.nu_lattice = ref_length/dx_coarse, nu_lattice_fine
    p.tau_coarse, p.tau_levels = tau_levels[1], tau_levels
    p.cs2, p.cs4 = 1f0/3f0, 1f0/9f0
    p.re_number, p.u_physical, p.rho_physical, p.nu_physical = re_number, u_phys, rho_phys, nu_phys
    p.length_scale, p.time_scale, p.velocity_scale, p.force_scale = length_scale, time_scale, velocity_scale, force_scale
    p.tau_fine, p.tau_margin_percent = tau_fine, (tau_fine - 0.5)/0.5*100
    p.wall_model_active, p.y_plus_first_cell = WALL_MODEL_ENABLED, 0.0
    p.estimated_memory_gb = estimated_memory_gb
    
    return p
end

function print_domain_summary()
    !DOMAIN_PARAMS.initialized && (println("Domain not initialized."); return)
    p = DOMAIN_PARAMS
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║                 LBM DOMAIN CONFIGURATION SUMMARY                 ║")
    println("╠══════════════════════════════════════════════════════════════════╣")
    @printf("║  Case: %-40s      ║\n", basename(CASE_DIR))
    @printf("║  Re = %-12.0f | %d levels | Est. %.2f GB                    ║\n", p.re_number, p.num_levels, p.estimated_memory_gb)
    println("╚══════════════════════════════════════════════════════════════════╝")
end

get_domain_params() = DOMAIN_PARAMS.initialized ? DOMAIN_PARAMS : error("Domain not initialized")
get_num_levels() = DOMAIN_PARAMS.initialized ? DOMAIN_PARAMS.num_levels : NUM_LEVELS_CONFIG