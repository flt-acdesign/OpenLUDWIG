// # FILE: .\src\bouzidi_setup.jl
using Base.Threads
using StaticArrays

"""
BOUZIDI_SETUP.JL - Preprocessing and Initialization
"""

"""
    build_block_triangle_map_for_bouzidi - Build spatial hash of triangles per block
"""
function build_block_triangle_map_for_bouzidi(mesh, 
                                              sorted_blocks::Vector{Tuple{Int,Int,Int}},
                                              dx::Float64,
                                              mesh_offset::SVector{3,Float64},
                                              block_size::Int)
    block_tris = [Int[] for _ in 1:length(sorted_blocks)]
    bs = block_size
    
    b_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        b_lookup[coord] = i
    end
    
    margin = dx * 2.5
    
    for (t_idx, tri) in enumerate(mesh.triangles)
        v1 = SVector{3,Float64}(tri[1])
        v2 = SVector{3,Float64}(tri[2])
        v3 = SVector{3,Float64}(tri[3])
        
        tri_min = min.(v1, min.(v2, v3))
        tri_max = max.(v1, max.(v2, v3))
        
        min_pt = tri_min + mesh_offset
        max_pt = tri_max + mesh_offset
        
        min_b = floor.(Int, (min_pt .- margin) ./ (bs * dx)) .+ 1
        max_b = floor.(Int, (max_pt .+ margin) ./ (bs * dx)) .+ 1
        
        for bz in max(1, min_b[3]):max_b[3]
            for by in max(1, min_b[2]):max_b[2]
                for bx in max(1, min_b[1]):max_b[1]
                    if haskey(b_lookup, (bx, by, bz))
                        idx = b_lookup[(bx, by, bz)]
                        push!(block_tris[idx], t_idx)
                    end
                end
            end
        end
    end
    
    return block_tris
end

"""
    compute_bouzidi_qmap_sparse(level_active_coords, mesh, dx, mesh_offset, block_size)

Compute Q-values and build sparse boundary cell coordinate lists.
Also computes the mapping from boundary links to triangle indices.

Returns: (q_map, cell_block, cell_x, cell_y, cell_z, tri_map, n_boundary_cells)
"""
function compute_bouzidi_qmap_sparse(level_active_coords::Vector{Tuple{Int,Int,Int}},
                                     mesh,
                                     dx::Float64,
                                     mesh_offset::SVector{3,Float64},
                                     block_size::Int)
    
    println("[Bouzidi] Computing Q-values and surface mapping for $(length(level_active_coords)) blocks...")
    
    n_blocks = length(level_active_coords)
    bs = block_size
    
    cx = Int32[]
    cy = Int32[]
    cz = Int32[]
    for dz in -1:1, dy in -1:1, dx_i in -1:1
        push!(cx, Int32(dx_i))
        push!(cy, Int32(dy))
        push!(cz, Int32(dz))
    end
    
    q_map_cpu = zeros(Float16, bs, bs, bs, n_blocks, 27)
    tri_map_cpu = zeros(Int32, bs, bs, bs, n_blocks, 27)
    
    max_tid = Threads.nthreads()
    if isdefined(Threads, :threadpoolsize)
        try
            max_tid += Threads.threadpoolsize(:interactive)
        catch
        end
    end
    
    thread_boundary_cells = [Vector{NTuple{4,Int32}}() for _ in 1:max_tid]
    
    block_tri_map = build_block_triangle_map_for_bouzidi(mesh, level_active_coords, 
                                                         dx, mesh_offset, bs)
    
    @threads for b_idx in 1:n_blocks
        tid = Threads.threadid()
        (bx, by, bz) = level_active_coords[b_idx]
        relevant_tris_indices = block_tri_map[b_idx]
        
        if isempty(relevant_tris_indices)
            continue
        end

        if tid > length(thread_boundary_cells)
             continue 
        end
        
        local_triangles = [mesh.triangles[t_idx] for t_idx in relevant_tris_indices]
        
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            px = ((bx - 1) * bs + lx - 0.5) * dx
            py = ((by - 1) * bs + ly - 0.5) * dx
            pz = ((bz - 1) * bs + lz - 0.5) * dx
            cell_center = SVector(px, py, pz)
            
            # Now returns Tuple(q_vals, tri_indices)
            q_vals, tri_idxs = compute_q_for_cell(cell_center, dx, local_triangles, 
                                                  mesh_offset, cx, cy, cz, relevant_tris_indices)
            
            has_boundary = false
            for k in 1:27
                if q_vals[k] > 0.0
                    q_map_cpu[lx, ly, lz, b_idx, k] = Float16(q_vals[k])
                    tri_map_cpu[lx, ly, lz, b_idx, k] = tri_idxs[k]
                    has_boundary = true
                end
            end
            
            if has_boundary
                push!(thread_boundary_cells[tid], (Int32(b_idx), Int32(lx), Int32(ly), Int32(lz)))
            end
        end
    end
    
    all_boundary_cells = NTuple{4,Int32}[]
    for cells in thread_boundary_cells
        append!(all_boundary_cells, cells)
    end
    
    n_boundary = length(all_boundary_cells)
    total_cells = n_blocks * bs^3
    
    cell_block = Vector{Int32}(undef, n_boundary)
    cell_x = Vector{Int32}(undef, n_boundary)
    cell_y = Vector{Int32}(undef, n_boundary)
    cell_z = Vector{Int32}(undef, n_boundary)
    
    for (i, (b, x, y, z)) in enumerate(all_boundary_cells)
        cell_block[i] = b
        cell_x[i] = x
        cell_y[i] = y
        cell_z[i] = z
    end
    
    qmap_mem_mb = sizeof(q_map_cpu) / 1024^2
    sparse_mem_kb = (sizeof(cell_block) + sizeof(cell_x) + sizeof(cell_y) + sizeof(cell_z)) / 1024
    
    println("[Bouzidi] Found $n_boundary boundary cells ($(round(100*n_boundary/total_cells, digits=2))%)")
    println("[Bouzidi] Q-map: $(round(qmap_mem_mb, digits=1)) MB, Sparse coords: $(round(sparse_mem_kb, digits=1)) KB")
    
    return q_map_cpu, cell_block, cell_x, cell_y, cell_z, tri_map_cpu, n_boundary
end