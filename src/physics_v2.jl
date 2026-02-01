# FILE: ./src/physics_v2.jl
"""
PHYSICS_V2.JL - LBM Solver Orchestration

This file orchestrates the physics calculations by loading the necessary
math libraries and kernels, and providing the high-level timestep function.
"""

using KernelAbstractions
using StaticArrays
using CUDA
using Adapt

# CONSTANTS
const KAPPA = 0.41f0
const CS2_PHYSICS = 1.0f0 / 3.0f0
const CS4_PHYSICS = CS2_PHYSICS * CS2_PHYSICS


# INCLUDES
# Note: physics_utils.jl now includes gradients
include("physics_utils.jl")
include("physics_interpolation.jl")
include("physics_kernels.jl")

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
    temporal_weight::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool
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
    
    kernel! = stream_collide_kernel_v2!(backend)
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