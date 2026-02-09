// # FILE: .\src\domain.jl
"""
DOMAIN.JL - Multi-Level Block Domain Orchestration

Orchestrates the creation of the simulation domain by utilizing
topology and generation modules.
"""

include("blocks.jl")
include("bouzidi.jl")

using Base.Threads
using LinearAlgebra
using StaticArrays
using Printf

include("domain_topology.jl")
include("domain_generation.jl")

function setup_multilevel_domain(mesh::Geometry.SolverMesh, params)
    println("\n" * "="^70)
    println("[Setup] Multi-Level Domain Generation")
    if REFINEMENT_STRATEGY == :geometry_first
        println("        Strategy: Geometry-First (Direct Mesh Check)")
    else
        println("        Strategy: Topology-Legacy (Inherited Obstacles)")
    end
    if BOUNDARY_METHOD == :bouzidi
        println("        Boundary: Bouzidi IBM (finest $BOUZIDI_LEVELS level(s))")
    else
        println("        Boundary: Simple Bounce-Back")
    end
    println("        Temporal Interpolation: $(TEMPORAL_INTERPOLATION ? "ENABLED" : "DISABLED")")
    println("="^70)
    
    num_levels = params.num_levels
    mesh_offset = SVector(params.mesh_offset[1], params.mesh_offset[2], params.mesh_offset[3])
    
    placed_mesh_min = SVector(params.mesh_min...) + mesh_offset
    placed_mesh_max = SVector(params.mesh_max...) + mesh_offset
    
    wake_start_x = placed_mesh_max[1] - (params.reference_length * 0.1)
    wake_end_x = placed_mesh_max[1] + (params.reference_length * WAKE_REFINEMENT_LENGTH)
    
    wake_center_y = (placed_mesh_min[2] + placed_mesh_max[2]) / 2.0
    wake_center_z = (placed_mesh_min[3] + placed_mesh_max[3]) / 2.0
    
    wake_width = (placed_mesh_max[2] - placed_mesh_min[2]) * WAKE_REFINEMENT_WIDTH_FACTOR
    wake_height = (placed_mesh_max[3] - placed_mesh_min[3]) * WAKE_REFINEMENT_HEIGHT_FACTOR
    
    wake_min_y = wake_center_y - wake_width/2.0
    wake_max_y = wake_center_y + wake_width/2.0
    wake_min_z = wake_center_z - wake_height/2.0
    wake_max_z = wake_center_z + wake_height/2.0
    
    grids = Vector{BlockLevel}()
    
    for lvl in 1:num_levels
        println("\n--- Level $lvl ---")
        
        scale = 2^(lvl - 1)
        dx = params.dx_coarse / scale
        dt = 1.0f0 / Float32(scale)
        tau = params.tau_levels[lvl]
        
        bx_max = params.bx_max * scale
        by_max = params.by_max * scale
        bz_max = params.bz_max * scale
        
        active_set = Set{Tuple{Int,Int,Int}}()
        
        if lvl == 1
            println("  Generating Full Wind Tunnel for Level 1...")
            for bz in 1:bz_max, by in 1:by_max, bx in 1:bx_max
                push!(active_set, (bx, by, bz))
            end
            println("  Full domain: $(bx_max)×$(by_max)×$(bz_max) = $(length(active_set)) blocks")
        else
            prev_level = grids[lvl-1]
            prev_scale = 2^(lvl - 2)
            prev_dx = params.dx_coarse / prev_scale
            prev_bs_phys = BLOCK_SIZE * prev_dx
            
            if REFINEMENT_STRATEGY == :geometry_first
                surface_blocks = get_active_blocks_for_level(mesh, dx, mesh_offset, bx_max, by_max, bz_max)
                union!(active_set, surface_blocks)
                
                if ENABLE_WAKE_REFINEMENT
                    for (cbx, cby, cbz) in prev_level.active_block_coords
                        b_min_x = (cbx - 1) * prev_bs_phys
                        b_max_x = cbx * prev_bs_phys
                        b_min_y = (cby - 1) * prev_bs_phys
                        b_max_y = cby * prev_bs_phys
                        b_min_z = (cbz - 1) * prev_bs_phys
                        b_max_z = cbz * prev_bs_phys
                        
                        overlap_x = (b_min_x <= wake_end_x) && (b_max_x >= wake_start_x)
                        overlap_y = (b_min_y <= wake_max_y) && (b_max_y >= wake_min_y)
                        overlap_z = (b_min_z <= wake_max_z) && (b_max_z >= wake_min_z)
                        
                        if overlap_x && overlap_y && overlap_z
                            for dbz in 0:1, dby in 0:1, dbx in 0:1
                                fbx = 2*cbx - 1 + dbx
                                fby = 2*cby - 1 + dby
                                fbz = 2*cbz - 1 + dbz
                                if fbx >= 1 && fbx <= bx_max && fby >= 1 && fby <= by_max && fbz >= 1 && fbz <= bz_max
                                    push!(active_set, (fbx, fby, fbz))
                                end
                            end
                        end
                    end
                end
                
                prev_level_set = Set(grids[lvl-1].active_block_coords)
                orphans = 0
                final_set = Set{Tuple{Int,Int,Int}}()
                for (bx, by, bz) in active_set
                    pbx = (bx + 1) ÷ 2
                    pby = (by + 1) ÷ 2
                    pbz = (bz + 1) ÷ 2
                    if (pbx, pby, pbz) in prev_level_set
                        push!(final_set, (bx, by, bz))
                    else
                        orphans += 1
                    end
                end
                active_set = final_set
            else
                for (b_idx, (cbx, cby, cbz)) in enumerate(prev_level.active_block_coords)
                    is_surface = any(view(prev_level.obstacle, :, :, :, b_idx))
                    
                    is_wake = false
                    if ENABLE_WAKE_REFINEMENT && !is_surface
                        b_min_x = (cbx - 1) * prev_bs_phys
                        b_max_x = cbx * prev_bs_phys
                        b_min_y = (cby - 1) * prev_bs_phys
                        b_max_y = cby * prev_bs_phys
                        b_min_z = (cbz - 1) * prev_bs_phys
                        b_max_z = cbz * prev_bs_phys
                        
                        overlap_x = (b_min_x <= wake_end_x) && (b_max_x >= wake_start_x)
                        overlap_y = (b_min_y <= wake_max_y) && (b_max_y >= wake_min_y)
                        overlap_z = (b_min_z <= wake_max_z) && (b_max_z >= wake_min_z)
                        is_wake = overlap_x && overlap_y && overlap_z
                    end
                    
                    if is_surface || is_wake
                        for dbz in 0:1, dby in 0:1, dbx in 0:1
                            fbx = 2*cbx - 1 + dbx
                            fby = 2*cby - 1 + dby
                            fbz = 2*cbz - 1 + dbz
                            if fbx >= 1 && fbx <= bx_max && fby >= 1 && fby <= by_max && fbz >= 1 && fbz <= bz_max
                                push!(active_set, (fbx, fby, fbz))
                            end
                        end
                    end
                end
            end
        end
        
        n_before_halo = length(active_set)
        
        add_halo_blocks_with_siblings!(active_set, REFINEMENT_MARGIN, bx_max, by_max, bz_max)
        ensure_complete_parent_coverage!(active_set, bx_max, by_max, bz_max)
        
        n_after_halo = length(active_set)
        if lvl > 1
            println("  Added $(n_after_halo - n_before_halo) halo blocks")
        end
        
        sorted_blocks = sort(collect(active_set))
        n_blocks = length(sorted_blocks)
        
        nb_table = build_neighbor_table(sorted_blocks, bx_max, by_max, bz_max)
        
        bs = BLOCK_SIZE
        obstacle_arr = falses(bs, bs, bs, n_blocks)
        sponge_arr = zeros(Float32, bs, bs, bs, n_blocks)
        wall_dist_arr = fill(100.0f0, bs, bs, bs, n_blocks)
        
        block_ptr = zeros(Int32, bx_max, by_max, bz_max)
        for (idx, (bx, by, bz)) in enumerate(sorted_blocks)
            block_ptr[bx, by, bz] = Int32(idx)
        end
        
        voxelize_blocks!(obstacle_arr, sorted_blocks, mesh, dx, mesh_offset)
        perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, bx_max, by_max, bz_max)
        
        apply_sponge!(sponge_arr, sorted_blocks, params, scale)
        
        if WALL_MODEL_ENABLED
            compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, mesh, dx, mesh_offset)
        end
        
        use_bouzidi = should_use_bouzidi(lvl, num_levels, BOUNDARY_METHOD, BOUZIDI_LEVELS)
        
        bouzidi_q_map = nothing
        bouzidi_cell_block = nothing
        bouzidi_cell_x = nothing
        bouzidi_cell_y = nothing
        bouzidi_cell_z = nothing
        bouzidi_tri_map = nothing
        n_boundary_cells = 0
        
        if use_bouzidi
            println("[Bouzidi] Computing Q-map and surface mapping for Level $lvl...")
            
            # Now returns tri_map as well
            q_map_cpu, cell_block, cell_x, cell_y, cell_z, tri_map, n_b = 
                compute_bouzidi_qmap_sparse(sorted_blocks, mesh, dx, mesh_offset, BLOCK_SIZE)
            
            bouzidi_q_map = q_map_cpu
            bouzidi_cell_block = cell_block
            bouzidi_cell_x = cell_x
            bouzidi_cell_y = cell_y
            bouzidi_cell_z = cell_z
            bouzidi_tri_map = tri_map
            n_boundary_cells = n_b
            
            println("[Bouzidi] Level $lvl: $n_boundary_cells boundary cells")
        end
        
        
        level = BlockLevel(lvl, sorted_blocks, nb_table, Float32(dx), dt, tau;
                           bouzidi_q_map = bouzidi_q_map,
                           bouzidi_cell_block = bouzidi_cell_block,
                           bouzidi_cell_x = bouzidi_cell_x,
                           bouzidi_cell_y = bouzidi_cell_y,
                           bouzidi_cell_z = bouzidi_cell_z,
                           bouzidi_tri_map = bouzidi_tri_map,
                           n_boundary_cells = n_boundary_cells,
                           enable_temporal_interpolation = TEMPORAL_INTERPOLATION)
        
        level.obstacle .= obstacle_arr
        level.sponge .= sponge_arr
        level.wall_dist .= wall_dist_arr
        
        push!(grids, level)
        
        n_voxels = n_blocks * BLOCK_SIZE^3
        @printf("  Total: %d blocks, %.2f M voxels\n", n_blocks, n_voxels/1e6)
        if use_bouzidi
            println("  Boundary: Bouzidi IBM ($n_boundary_cells cells, sparse)")
        else
            println("  Boundary: Simple bounce-back")
        end
    end
    
    println("\n[Verify] Checking parent coverage...")
    for lvl in 2:num_levels
        fine_set = Set(grids[lvl].active_block_coords)
        coarse_set = Set(grids[lvl-1].active_block_coords)
        missing = 0
        for (fbx, fby, fbz) in fine_set
            pbx = (fbx + 1) ÷ 2
            pby = (fby + 1) ÷ 2
            pbz = (fbz + 1) ÷ 2
            if !((pbx, pby, pbz) in coarse_set)
                missing += 1
            end
        end
        println("  Level $lvl: $(missing == 0 ? "✓" : "⚠") (missing parents: $missing)")
    end
    
    return grids
end

function setup_multilevel_domain(stl_path::String; num_levels=NUM_LEVELS)
    println("[Domain] Loading: $stl_path")
    if !isfile(stl_path); error("STL file not found: $stl_path"); end

    mesh = Geometry.load_mesh(stl_path, scale=Float64(STL_SCALE))
    bounds = Geometry.compute_mesh_bounds(mesh)

    compute_domain_from_mesh(Tuple(bounds.min_bounds), Tuple(bounds.max_bounds))
    params = get_domain_params()
    print_domain_summary()

    return setup_multilevel_domain(mesh, params), mesh
end

"""
    setup_multilevel_domain_from_parts(parts; num_levels)

Set up the multi-level domain from multiple geometry parts.
Merges all parts at time=0.0 and builds the domain.

Returns (grids, merged_mesh, params).
"""
function setup_multilevel_domain_from_parts(parts::Vector{Geometry.GeometryPart}; num_levels=NUM_LEVELS)
    mesh = Geometry.merge_geometry_parts(parts, 0.0)
    bounds = Geometry.compute_mesh_bounds(mesh)

    compute_domain_from_mesh(Tuple(bounds.min_bounds), Tuple(bounds.max_bounds))
    params = get_domain_params()
    print_domain_summary()

    return setup_multilevel_domain(mesh, params), mesh
end

"""
    revoxelize_geometry!(cpu_grids, mesh, params)

Re-voxelize the obstacle array and recompute Bouzidi q-maps for all grid levels.
Used when geometry has moved (dynamic rotation).

This updates the CPU grid arrays in place. The caller must then transfer
the updated arrays to GPU.

Arguments:
- cpu_grids: Vector of BlockLevel (CPU side)
- mesh: Updated SolverMesh with moved geometry
- params: Domain parameters
"""
function revoxelize_geometry!(cpu_grids::Vector, mesh::Geometry.SolverMesh, params)
    num_levels = length(cpu_grids)
    mesh_offset = SVector(params.mesh_offset[1], params.mesh_offset[2], params.mesh_offset[3])

    println("[Remesh] Re-voxelizing geometry for $(num_levels) levels...")

    for lvl in 1:num_levels
        level = cpu_grids[lvl]
        sorted_blocks = level.active_block_coords
        n_blocks = length(sorted_blocks)
        scale = 2^(lvl - 1)
        dx = params.dx_coarse / scale

        bs = BLOCK_SIZE

        # Re-voxelize: clear and recompute obstacle array
        obstacle_arr = falses(bs, bs, bs, n_blocks)
        voxelize_blocks!(obstacle_arr, sorted_blocks, mesh, dx, mesh_offset)

        # Re-do flood fill
        bx_max = level.grid_dim_x
        by_max = level.grid_dim_y
        bz_max = level.grid_dim_z
        block_ptr = zeros(Int32, bx_max, by_max, bz_max)
        for (idx, (bx, by, bz)) in enumerate(sorted_blocks)
            block_ptr[bx, by, bz] = Int32(idx)
        end
        perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, bx_max, by_max, bz_max)

        # Update obstacle in the CPU grid
        level.obstacle .= obstacle_arr

        # Recompute wall distances if wall model is enabled
        if WALL_MODEL_ENABLED
            wall_dist_arr = fill(100.0f0, bs, bs, bs, n_blocks)
            compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, mesh, dx, mesh_offset)
            level.wall_dist .= wall_dist_arr
        end

        # Recompute Bouzidi q-maps if this level uses Bouzidi
        use_bouzidi = should_use_bouzidi(lvl, num_levels, BOUNDARY_METHOD, BOUZIDI_LEVELS)

        if use_bouzidi
            q_map_cpu, cell_block, cell_x, cell_y, cell_z, tri_map, n_b =
                compute_bouzidi_qmap_sparse(sorted_blocks, mesh, dx, mesh_offset, BLOCK_SIZE)

            # Update Bouzidi data in the level
            if n_b > 0
                level.bouzidi_q_map .= 0
                # Copy new q-map values (sizes should match since block structure hasn't changed)
                q_map_f16 = convert(Array{Float16, 5}, q_map_cpu)
                tri_map_i32 = convert(Array{Int32, 5}, tri_map)

                if size(level.bouzidi_q_map) == size(q_map_f16)
                    level.bouzidi_q_map .= q_map_f16
                    level.bouzidi_tri_map .= tri_map_i32
                end
            end

            println("[Remesh] Level $lvl: $n_b boundary cells (Bouzidi)")
        end
    end

    println("[Remesh] Re-voxelization complete")
end

"""
    update_gpu_geometry!(grids, cpu_grids, backend)

Transfer updated obstacle and Bouzidi arrays from CPU to GPU after re-voxelization.

Arguments:
- grids: Vector of GPU BlockLevel (to update)
- cpu_grids: Vector of CPU BlockLevel (source of updated data)
- backend: KernelAbstractions backend
"""
function update_gpu_geometry!(grids, cpu_grids, backend)
    for lvl in 1:length(grids)
        # Update obstacle array on GPU
        copyto!(grids[lvl].obstacle, cpu_grids[lvl].obstacle)

        # Update wall distances
        if WALL_MODEL_ENABLED
            copyto!(grids[lvl].wall_dist, cpu_grids[lvl].wall_dist)
        end

        # Update Bouzidi q-map if applicable
        if grids[lvl].bouzidi_enabled
            copyto!(grids[lvl].bouzidi_q_map, cpu_grids[lvl].bouzidi_q_map)
            copyto!(grids[lvl].bouzidi_tri_map, cpu_grids[lvl].bouzidi_tri_map)
        end
    end
end