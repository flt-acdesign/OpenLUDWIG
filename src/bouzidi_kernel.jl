// # FILE: .\src\bouzidi_kernel.jl
using KernelAbstractions
using CUDA

"""
BOUZIDI_KERNEL.JL - GPU Correction Kernel

The Bouzidi interpolation scheme:
- For q < 0.5:  f_opp(x_f, t+1) = 2q * f_k(x_f, t*) + (1-2q) * f_k(x_ff, t*)
- For q >= 0.5: f_opp(x_f, t+1) = (1/2q) * f_k(x_f, t*) + (2q-1)/(2q) * f_opp(x_f, t*)
"""

@kernel function bouzidi_correction_kernel_fixed!(
    f_out,                
    f_post_collision,     
    q_map,                
    cell_block,           
    cell_x,               
    cell_y,               
    cell_z,               
    neighbor_table,       
    block_size::Int32,
    cx_arr, cy_arr, cz_arr,
    opp_arr,
    q_min_threshold::Float32
)
    cell_idx = @index(Global)
    
    @inbounds begin
        b_idx = cell_block[cell_idx]
        x = cell_x[cell_idx]
        y = cell_y[cell_idx]
        z = cell_z[cell_idx]
        
        for k in 1:27
            q = Float32(q_map[x, y, z, b_idx, k])
            
            if q > q_min_threshold && q <= 1.0f0
                opp_k = opp_arr[k]
                
                # f_k at fluid node x_f (post-collision)
                f_k = f_post_collision[x, y, z, b_idx, k]
                
                if q < 0.5f0
                    
                    # This is x_ff: the cell "behind" x_f
                    ncx = cx_arr[opp_k]
                    ncy = cy_arr[opp_k]
                    ncz = cz_arr[opp_k]
                    
                    nx = x + ncx
                    ny = y + ncy
                    nz = z + ncz
                    
                    f_ff = f_k  # Default fallback
                    
                    if nx >= 1 && nx <= block_size && ny >= 1 && ny <= block_size && nz >= 1 && nz <= block_size
                        f_ff = f_post_collision[nx, ny, nz, b_idx, k]
                    else
                        nb_off_x = (nx < 1) ? Int32(-1) : (nx > block_size ? Int32(1) : Int32(0))
                        nb_off_y = (ny < 1) ? Int32(-1) : (ny > block_size ? Int32(1) : Int32(0))
                        nb_off_z = (nz < 1) ? Int32(-1) : (nz > block_size ? Int32(1) : Int32(0))
                        
                        dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) + (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
                        nb_block = neighbor_table[b_idx, dir_idx]
                        
                        if nb_block > 0
                            nnx = nx < 1 ? nx + block_size : (nx > block_size ? nx - block_size : nx)
                            nny = ny < 1 ? ny + block_size : (ny > block_size ? ny - block_size : ny)
                            nnz = nz < 1 ? nz + block_size : (nz > block_size ? nz - block_size : nz)
                            f_ff = f_post_collision[nnx, nny, nnz, nb_block, k]
                        end
                    end
                    
                    # Interpolation for q < 0.5
                    coeff1 = 2.0f0 * q
                    f_out[x, y, z, b_idx, opp_k] = coeff1 * f_k + (1.0f0 - coeff1) * f_ff
                    
                else
                    
                    # Interpolation for q >= 0.5
                    f_opp_post = f_post_collision[x, y, z, b_idx, opp_k]
                    
                    inv_2q = 1.0f0 / (2.0f0 * q)
                    coeff2 = (2.0f0 * q - 1.0f0) * inv_2q
                    
                    f_out[x, y, z, b_idx, opp_k] = inv_2q * f_k + coeff2 * f_opp_post
                end
            end
        end
    end
end

"""
    apply_bouzidi_correction!(f_out, f_post_collision, ...)

Apply Bouzidi boundary correction. Call AFTER the main stream-collide kernel.
"""
function apply_bouzidi_correction!(f_out, f_post_collision,
                                   q_map, cell_block, cell_x, cell_y, cell_z, n_boundary_cells,
                                   neighbor_table,
                                   block_size::Int,
                                   cx_gpu, cy_gpu, cz_gpu, opp_gpu,
                                   q_min_threshold::Float32,
                                   backend)
    
    if n_boundary_cells == 0
        return
    end
    
    kernel! = bouzidi_correction_kernel_fixed!(backend)
    kernel!(f_out, f_post_collision,
            q_map,
            cell_block,
            cell_x,
            cell_y,
            cell_z,
            neighbor_table,
            Int32(block_size),
            cx_gpu, cy_gpu, cz_gpu, opp_gpu,
            q_min_threshold,
            ndrange=(n_boundary_cells,))
end