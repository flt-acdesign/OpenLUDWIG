# FILE: ./src/physics_utils.jl
using KernelAbstractions

"""
PHYSICS_UTILS.JL - Low-level Math Helpers, Equilibrium, and Gradients

Consolidated:
1. Hashing and Random Noise
2. Equilibrium Calculation
3. Velocity Gradient Calculation (Finite Difference)
"""

# ==============================================================================
# HASHING & NOISE
# ==============================================================================

@inline function gpu_hash(x::Int32)::UInt32
    h = reinterpret(UInt32, x)
    h = (h ⊻ (h >> 16)) * UInt32(0x85ebca6b)
    h = (h ⊻ (h >> 13)) * UInt32(0xc2b2ae35)
    return h ⊻ (h >> 16)
end

@inline function gradient_noise(gx::Int32, gy::Int32, gz::Int32, seed::Int32)::Float32
    combined = gx * Int32(374761393) + gy * Int32(668265263) + gz * Int32(1274126177) + seed
    h = gpu_hash(combined)
    return (Float32(h & UInt32(0xFFFF)) / 32768.0f0) - 1.0f0
end

# ==============================================================================
# EQUILIBRIUM
# ==============================================================================

@inline function calculate_equilibrium(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                       w_k::Float32, cx::Float32, cy::Float32, cz::Float32)
    cu = cx*ux + cy*uy + cz*uz
    usq = ux*ux + uy*uy + uz*uz
    return rho * w_k * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - 1.5f0*usq)
end

# ==============================================================================
# VELOCITY GRADIENTS
# ==============================================================================

@inline function get_velocity_neighbor(vel_in, x, y, z, b_idx, dx, dy, dz, block_size, neighbor_table)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)
    
    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        return vel_in[nx, ny, nz, ib, 1], vel_in[nx, ny, nz, ib, 2], vel_in[nx, ny, nz, ib, 3]
    end
    
    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]
    
    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        return vel_in[nnx, nny, nnz, nb_idx, 1], vel_in[nnx, nny, nnz, nb_idx, 2], vel_in[nnx, nny, nnz, nb_idx, 3]
    end
    
    return vel_in[x, y, z, ib, 1], vel_in[x, y, z, ib, 2], vel_in[x, y, z, ib, 3]
end

@inline function compute_velocity_gradients(vel_in, x, y, z, b_idx, block_size, neighbor_table)
    ux_E, uy_E, uz_E = get_velocity_neighbor(vel_in, x, y, z, b_idx, 1, 0, 0, block_size, neighbor_table)
    ux_W, uy_W, uz_W = get_velocity_neighbor(vel_in, x, y, z, b_idx, -1, 0, 0, block_size, neighbor_table)
    ux_N, uy_N, uz_N = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, 1, 0, block_size, neighbor_table)
    ux_S, uy_S, uz_S = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, -1, 0, block_size, neighbor_table)
    ux_T, uy_T, uz_T = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, 0, 1, block_size, neighbor_table)
    ux_B, uy_B, uz_B = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, 0, -1, block_size, neighbor_table)
    
    return (0.5f0*(ux_E-ux_W), 0.5f0*(ux_N-ux_S), 0.5f0*(ux_T-ux_B),
            0.5f0*(uy_E-uy_W), 0.5f0*(uy_N-uy_S), 0.5f0*(uy_T-uy_B),
            0.5f0*(uz_E-uz_W), 0.5f0*(uz_N-uz_S), 0.5f0*(uz_T-uz_B))
end