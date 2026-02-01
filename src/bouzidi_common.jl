// # FILE: .\src\bouzidi_common.jl
using Adapt
using Printf

"""
BOUZIDI_COMMON.JL - Data Structures and Utilities
"""

"""
    BouzidiDataSparse

Sparse storage for Bouzidi boundary conditions.
Now includes mapping to surface triangles.
"""
struct BouzidiDataSparse{A_Q, A_Idx, A_Tri}
    q_map::A_Q    
    cell_block::A_Idx    
    cell_x::A_Idx        
    cell_y::A_Idx        
    cell_z::A_Idx        
    tri_map::A_Tri
    n_boundary_cells::Int
end

"""
Determine if a level should use Bouzidi boundary conditions.
"""
function should_use_bouzidi(level_id::Int, num_levels::Int, 
                            boundary_method::Symbol, bouzidi_levels::Int)
    if boundary_method != :bouzidi
        return false
    end
    return level_id > (num_levels - bouzidi_levels)
end

"""
Transfer Bouzidi data to GPU.
"""
function create_bouzidi_data_gpu(q_map_cpu, cell_block, cell_x, cell_y, cell_z, tri_map_cpu,
                                 n_boundary::Int, backend)
    return BouzidiDataSparse(
        adapt(backend, q_map_cpu),
        adapt(backend, cell_block),
        adapt(backend, cell_x),
        adapt(backend, cell_y),
        adapt(backend, cell_z),
        adapt(backend, tri_map_cpu),
        n_boundary
    )
end

"""
Print Bouzidi memory usage summary.
"""
function print_bouzidi_memory(bouzidi_data::BouzidiDataSparse, level_id::Int)
    q_mem = sizeof(bouzidi_data.q_map) / 1024^2
    tri_mem = sizeof(bouzidi_data.tri_map) / 1024^2
    sparse_mem = (sizeof(bouzidi_data.cell_block) + sizeof(bouzidi_data.cell_x) + 
                  sizeof(bouzidi_data.cell_y) + sizeof(bouzidi_data.cell_z)) / 1024
    
    println("  Level $level_id Bouzidi: $(bouzidi_data.n_boundary_cells) cells, " *
            "$(round(q_mem, digits=1)) MB q_map, $(round(tri_mem, digits=1)) MB tri_map, $(round(sparse_mem, digits=1)) KB coords")
end