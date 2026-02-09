// # FILE: .\src\domain_topology.jl
using Base.Threads
using StaticArrays

"""
DOMAIN_TOPOLOGY.JL - Block Connectivity and Hierarchy Management
"""

function get_active_blocks_for_level(mesh::Geometry.SolverMesh, 
                                     dx::Float64, 
                                     mesh_offset::SVector{3,Float64},
                                     bx_max::Int, by_max::Int, bz_max::Int)
    
    active_set = Set{Tuple{Int, Int, Int}}()
    bs = BLOCK_SIZE
    
    margin = dx * 0.01 
    inv_bs_dx = 1.0 / (bs * dx)
    
    for tri in mesh.triangles
        t_min = tri[1] .+ mesh_offset
        t_max = tri[1] .+ mesh_offset
        
        for p in tri
            pm = p .+ mesh_offset
            t_min = min.(t_min, pm)
            t_max = max.(t_max, pm)
        end
        
        min_bx = floor(Int, (t_min[1] - margin) * inv_bs_dx) + 1
        min_by = floor(Int, (t_min[2] - margin) * inv_bs_dx) + 1
        min_bz = floor(Int, (t_min[3] - margin) * inv_bs_dx) + 1
        
        max_bx = floor(Int, (t_max[1] + margin) * inv_bs_dx) + 1
        max_by = floor(Int, (t_max[2] + margin) * inv_bs_dx) + 1
        max_bz = floor(Int, (t_max[3] + margin) * inv_bs_dx) + 1
        
        min_bx = max(1, min_bx); max_bx = min(max_bx, bx_max)
        min_by = max(1, min_by); max_by = min(max_by, by_max)
        min_bz = max(1, min_bz); max_bz = min(max_bz, bz_max)
        
        for bz in min_bz:max_bz
            for by in min_by:max_by
                for bx in min_bx:max_bx
                    push!(active_set, (bx, by, bz))
                end
            end
        end
    end
    
    return active_set
end

function add_halo_blocks_with_siblings!(active_set::Set{Tuple{Int, Int, Int}}, 
                                        layers::Int,
                                        bx_max::Int, by_max::Int, bz_max::Int)
    neighbor_offsets = Tuple{Int,Int,Int}[]
    for dz in -1:1, dy in -1:1, dx in -1:1
        if dx==0 && dy==0 && dz==0; continue; end
        push!(neighbor_offsets, (dx, dy, dz))
    end
    
    for _ in 1:layers
        new_blocks = Set{Tuple{Int, Int, Int}}()
        
        for (bx, by, bz) in active_set
            for (dx, dy, dz) in neighbor_offsets
                nbx, nby, nbz = bx+dx, by+dy, bz+dz
                if nbx >= 1 && nbx <= bx_max && nby >= 1 && nby <= by_max && nbz >= 1 && nbz <= bz_max
                    if !((nbx, nby, nbz) in active_set)
                        push!(new_blocks, (nbx, nby, nbz))
                    end
                end
            end
        end
        
        siblings_to_add = Set{Tuple{Int, Int, Int}}()
        for (bx, by, bz) in new_blocks
            pbx = (bx + 1) ÷ 2
            pby = (by + 1) ÷ 2
            pbz = (bz + 1) ÷ 2
            
            for dbz in 0:1, dby in 0:1, dbx in 0:1
                sbx = 2*pbx - 1 + dbx
                sby = 2*pby - 1 + dby
                sbz = 2*pbz - 1 + dbz
                
                if sbx >= 1 && sbx <= bx_max && sby >= 1 && sby <= by_max && sbz >= 1 && sbz <= bz_max
                    if !((sbx, sby, sbz) in active_set) && !((sbx, sby, sbz) in new_blocks)
                        push!(siblings_to_add, (sbx, sby, sbz))
                    end
                end
            end
        end
        
        union!(active_set, new_blocks)
        union!(active_set, siblings_to_add)
    end
end

function ensure_complete_parent_coverage!(active_set::Set{Tuple{Int, Int, Int}},
                                          bx_max::Int, by_max::Int, bz_max::Int)
    added = true
    iterations = 0
    max_iterations = 10
    
    while added && iterations < max_iterations
        added = false
        iterations += 1
        siblings_to_add = Set{Tuple{Int, Int, Int}}()
        
        for (bx, by, bz) in active_set
            pbx = (bx + 1) ÷ 2
            pby = (by + 1) ÷ 2
            pbz = (bz + 1) ÷ 2
            
            for dbz in 0:1, dby in 0:1, dbx in 0:1
                sbx = 2*pbx - 1 + dbx
                sby = 2*pby - 1 + dby
                sbz = 2*pbz - 1 + dbz
                
                if sbx >= 1 && sbx <= bx_max && sby >= 1 && sby <= by_max && sbz >= 1 && sbz <= bz_max
                    if !((sbx, sby, sbz) in active_set)
                        push!(siblings_to_add, (sbx, sby, sbz))
                        added = true
                    end
                end
            end
        end
        
        union!(active_set, siblings_to_add)
    end
end

function build_neighbor_table(active_coords::Vector{Tuple{Int, Int, Int}}, 
                              bx_max::Int, by_max::Int, bz_max::Int)
    n_blocks = length(active_coords)
    table = zeros(Int32, n_blocks, 27)
    
    temp_ptr = zeros(Int32, bx_max, by_max, bz_max)
    for (i, (bx, by, bz)) in enumerate(active_coords)
        if bx >= 1 && bx <= bx_max && by >= 1 && by <= by_max && bz >= 1 && bz <= bz_max
            temp_ptr[bx, by, bz] = Int32(i)
        end
    end
    
    for i in 1:n_blocks
        (bx, by, bz) = active_coords[i]
        for dz in -1:1, dy in -1:1, dx in -1:1
            dir_idx = (dx+1) + (dy+1)*3 + (dz+1)*9 + 1
            nbx, nby, nbz = bx+dx, by+dy, bz+dz
            if nbx >= 1 && nbx <= bx_max && nby >= 1 && nby <= by_max && nbz >= 1 && nbz <= bz_max
                table[i, dir_idx] = temp_ptr[nbx, nby, nbz]
            else
                table[i, dir_idx] = 0
            end
        end
    end
    return table
end