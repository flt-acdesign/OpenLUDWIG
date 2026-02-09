// # FILE: .\src\blocks.jl
"""
BLOCKS.JL - Block-Level Data Structures with Temporal Interpolation Support

Added f_old, rho_old, vel_old arrays for temporal interpolation at refinement interfaces.
This enables smooth boundary conditions for fine grids instead of discontinuous jumps.
"""

using KernelAbstractions
using CUDA
using Adapt
using StaticArrays

const BLOCK_SIZE = 8

mutable struct BlockLevel{T_P4, T_P5, T_BlockPtr, T_NbTable, T_Obst, T_F16, T_I8, T_IntVec, T_MapVec, T_TriMap}
    level_id::Int
    dx::Float64
    dt::Float32
    tau::Float32

    grid_dim_x::Int
    grid_dim_y::Int
    grid_dim_z::Int

    block_pointer::T_BlockPtr
    active_block_coords::Vector{Tuple{Int, Int, Int}}

    # -- MACROSCOPIC VARS --
    rho::T_P4
    vel::T_P5
    vel_temp::T_P5

    # -- TEMPORAL INTERPOLATION STORAGE --
    rho_old::T_P4
    vel_old::T_P5

    # -- DISTRIBUTIONS --
    f::T_P5
    f_temp::T_P5
    f_post_collision::T_P5
    f_old::T_P5

    # -- THERMAL DDF (Phase 2) --
    tau_g::Float32
    g::T_P5
    g_temp::T_P5
    g_post_collision::T_P5
    g_old::T_P5
    temperature::T_P4
    temperature_old::T_P4

    # -- GEOMETRY / SPONGE --
    wall_dist::T_P4
    obstacle::T_Obst
    sponge::T_P4

    # -- CONNECTIVITY --
    neighbor_table::T_NbTable
    map_x::T_MapVec
    map_y::T_MapVec
    map_z::T_MapVec

    # -- BOUZIDI IBM DATA --
    bouzidi_enabled::Bool
    bouzidi_q_map::T_F16
    bouzidi_cell_block::T_IntVec
    bouzidi_cell_x::T_I8
    bouzidi_cell_y::T_I8
    bouzidi_cell_z::T_I8
    # MAPPING TO SURFACE TRIANGLES (New)
    bouzidi_tri_map::T_TriMap
    n_boundary_cells::Int
end

function Adapt.adapt_structure(to, level::BlockLevel)
    BlockLevel(
        level.level_id, level.dx, level.dt, level.tau,
        level.grid_dim_x, level.grid_dim_y, level.grid_dim_z,
        adapt(to, level.block_pointer),
        level.active_block_coords,
        adapt(to, level.rho), adapt(to, level.vel), adapt(to, level.vel_temp),
        adapt(to, level.rho_old), adapt(to, level.vel_old),
        adapt(to, level.f), adapt(to, level.f_temp),
        adapt(to, level.f_post_collision), adapt(to, level.f_old),
        level.tau_g,
        adapt(to, level.g), adapt(to, level.g_temp),
        adapt(to, level.g_post_collision), adapt(to, level.g_old),
        adapt(to, level.temperature), adapt(to, level.temperature_old),
        adapt(to, level.wall_dist), adapt(to, level.obstacle), adapt(to, level.sponge),
        adapt(to, level.neighbor_table),
        adapt(to, level.map_x), adapt(to, level.map_y), adapt(to, level.map_z),
        level.bouzidi_enabled,
        adapt(to, level.bouzidi_q_map),
        adapt(to, level.bouzidi_cell_block),
        adapt(to, level.bouzidi_cell_x), adapt(to, level.bouzidi_cell_y), adapt(to, level.bouzidi_cell_z),
        adapt(to, level.bouzidi_tri_map),
        level.n_boundary_cells
    )
end

function BlockLevel(level_id::Int,
                    active_coords::Vector{Tuple{Int, Int, Int}},
                    neighbor_table::Matrix{Int32},
                    dx::AbstractFloat, dt::AbstractFloat, tau::AbstractFloat;
                    bouzidi_q_map = nothing,
                    bouzidi_cell_block = nothing,
                    bouzidi_cell_x = nothing,
                    bouzidi_cell_y = nothing,
                    bouzidi_cell_z = nothing,
                    bouzidi_tri_map = nothing,
                    n_boundary_cells = 0,
                    enable_temporal_interpolation = true)

    n_blocks = length(active_coords)
    
    if isempty(active_coords)
        bx_max, by_max, bz_max = 0, 0, 0
    else
        bx_max = maximum(c[1] for c in active_coords)
        by_max = maximum(c[2] for c in active_coords)
        bz_max = maximum(c[3] for c in active_coords)
    end
    
    block_pointer = fill(Int32(0), bx_max, by_max, bz_max)
    for (i, (bx, by, bz)) in enumerate(active_coords)
        block_pointer[bx, by, bz] = Int32(i)
    end

    # --- Allocation ---
    rho = ones(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    vel = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    vel_temp = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    
    # Temporal interpolation
    if enable_temporal_interpolation && n_blocks > 0
        rho_old = ones(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        vel_old = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    else
        rho_old = ones(Float32, 1, 1, 1, 1)
        vel_old = zeros(Float32, 1, 1, 1, 1, 3)
    end
    
    # Distributions
    f = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
    f_temp = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
    
    if n_boundary_cells > 0
        f_post_collision = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
    else
        f_post_collision = zeros(Float32, 1, 1, 1, 1, 27)
    end
    
    if enable_temporal_interpolation && n_blocks > 0
        f_old = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
    else
        f_old = zeros(Float32, 1, 1, 1, 1, 27)
    end

    # Thermal DDF arrays (Phase 2)
    tau_g = 0.0f0
    if DDF_ENABLED && n_blocks > 0
        g = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
        g_temp = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
        temperature = fill(Float32(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        temperature_old = fill(Float32(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        if n_boundary_cells > 0
            g_post_collision = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
        else
            g_post_collision = zeros(Float32, 1, 1, 1, 1, 27)
        end
        if enable_temporal_interpolation
            g_old = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 27)
        else
            g_old = zeros(Float32, 1, 1, 1, 1, 27)
        end
    else
        g = zeros(Float32, 1, 1, 1, 1, 27)
        g_temp = zeros(Float32, 1, 1, 1, 1, 27)
        g_post_collision = zeros(Float32, 1, 1, 1, 1, 27)
        g_old = zeros(Float32, 1, 1, 1, 1, 27)
        temperature = ones(Float32, 1, 1, 1, 1)
        temperature_old = ones(Float32, 1, 1, 1, 1)
    end

    # Geometry arrays
    wall_dist = fill(100.0f0, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    obstacle = zeros(Bool, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    sponge = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    
    map_x = Int32[c[1] for c in active_coords]
    map_y = Int32[c[2] for c in active_coords]
    map_z = Int32[c[3] for c in active_coords]
    
    bouzidi_enabled = (n_boundary_cells > 0) && (bouzidi_q_map !== nothing)
    
    if bouzidi_enabled
        q_map_final = convert(Array{Float16, 5}, bouzidi_q_map)
        c_block_final = convert(Vector{Int32}, bouzidi_cell_block)
        c_x_final = convert(Vector{Int8}, bouzidi_cell_x)
        c_y_final = convert(Vector{Int8}, bouzidi_cell_y)
        c_z_final = convert(Vector{Int8}, bouzidi_cell_z)
        # Convert tri map to int32 
        tri_map_final = convert(Array{Int32, 5}, bouzidi_tri_map)
    else
        q_map_final = zeros(Float16, 1, 1, 1, 1, 27)
        c_block_final = Int32[]
        c_x_final = Int8[]
        c_y_final = Int8[]
        c_z_final = Int8[]
        tri_map_final = zeros(Int32, 1, 1, 1, 1, 27)
    end

    return BlockLevel(
        level_id, Float64(dx), Float32(dt), Float32(tau),
        bx_max, by_max, bz_max,
        block_pointer, active_coords,
        rho, vel, vel_temp,
        rho_old, vel_old,
        f, f_temp, f_post_collision, f_old,
        tau_g, g, g_temp, g_post_collision, g_old,
        temperature, temperature_old,
        wall_dist, obstacle, sponge,
        neighbor_table, map_x, map_y, map_z,
        bouzidi_enabled,
        q_map_final, c_block_final, c_x_final, c_y_final, c_z_final, tri_map_final,
        n_boundary_cells
    )
end

mutable struct SolverMesh
    levels::Vector{BlockLevel}
    domain_dims::Tuple{Int, Int, Int}
end

"""
Copy current state to old arrays for temporal interpolation.
Call BEFORE advancing the coarse level.
Optionally pass `g_current` for thermal DDF temporal storage.
"""
function copy_to_old!(level::BlockLevel, f_current, vel_current; g_current=nothing)
    if length(level.f_old) > 27
        copyto!(level.f_old, f_current)
        copyto!(level.rho_old, level.rho)
        copyto!(level.vel_old, vel_current)
    end
    # Thermal DDF temporal storage
    if g_current !== nothing && has_thermal(level) && length(level.g_old) > 27
        copyto!(level.g_old, g_current)
        copyto!(level.temperature_old, level.temperature)
    end
end

"""Check if level has temporal interpolation storage allocated."""
has_temporal_storage(level::BlockLevel) = length(level.f_old) > 27

"""Check if level has thermal DDF arrays allocated (not stubs)."""
has_thermal(level::BlockLevel) = length(level.g) > 27