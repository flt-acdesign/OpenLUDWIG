using KernelAbstractions



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


@inline function calculate_equilibrium(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                       w_k::Float32, cx::Float32, cy::Float32, cz::Float32)
    cu = cx*ux + cy*uy + cz*uz
    usq = ux*ux + uy*uy + uz*uz
    return rho * w_k * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - 1.5f0*usq)
end

# Compressibility-corrected equilibrium including 3rd and 4th order Hermite terms.
# On the D3Q27 lattice these terms are representable and extend the valid Mach
# number range from ~0.3 to ~0.6–0.9.
#
# Using the Hermite expansion (Coreixas et al. 2017, Shan et al. 2006):
# f_eq = ρ w_k [ 1 + (c·u)/cs² + ((c·u)² − cs²u²)/(2cs⁴)
#                + ((c·u)³ − 3cs²(c·u)u²)/(6cs⁶)
#                + ((c·u)⁴ − 6cs²(c·u)²u² + 3cs⁴u⁴)/(24cs⁸) ]
#
# Expanded with cs² = 1/3:
#   2nd order:  +3(c·u)  +4.5(c·u)²  −1.5u²
#   3rd order:  +4.5(c·u)³  −4.5(c·u)u²
#   4th order:  +3.375(c·u)⁴  −6.75(c·u)²u²  +1.125u⁴
@inline function calculate_equilibrium_compressible(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                                     w_k::Float32, cx::Float32, cy::Float32, cz::Float32)
    cu  = cx*ux + cy*uy + cz*uz
    usq = ux*ux + uy*uy + uz*uz
    cu2 = cu * cu
    cu3 = cu2 * cu
    cu4 = cu2 * cu2
    usq2 = usq * usq

    # Standard 2nd-order terms
    val = 1.0f0 + 3.0f0*cu + 4.5f0*cu2 - 1.5f0*usq

    # 3rd-order Hermite: ((c·u)³ − 3cs²(c·u)u²) / (6cs⁶)
    #   = 27/6 · (c·u)³ − 27/6 · (c·u)·u²  = 4.5·(c·u)³ − 4.5·(c·u)·u²
    val += 4.5f0*cu3 - 4.5f0*cu*usq

    # 4th-order Hermite: ((c·u)⁴ − 6cs²(c·u)²u² + 3cs⁴u⁴) / (24cs⁸)
    #   = 81/24·(c·u)⁴ − 6·81/(24·3)·(c·u)²u² + 3·81/(24·9)·u⁴
    #   = 3.375·(c·u)⁴ − 6.75·(c·u)²·u² + 1.125·u⁴
    val += 3.375f0*cu4 - 6.75f0*cu2*usq + 1.125f0*usq2

    return rho * w_k * val
end

# Dispatch: pick compressible or standard equilibrium based on flag
@inline function calculate_equilibrium_auto(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                             w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
                                             compressible::Int32)
    if compressible == Int32(1)
        return calculate_equilibrium_compressible(rho, ux, uy, uz, w_k, cx, cy, cz)
    else
        return calculate_equilibrium(rho, ux, uy, uz, w_k, cx, cy, cz)
    end
end


# Thermal equilibrium for the energy/temperature DDF (Phase 2).
# Advection-diffusion model: g_eq_k = w_k * T * (1 + 3*(c_k . u))
# Recovers: dT/dt + u.grad(T) = kappa * laplacian(T)
@inline function calculate_thermal_equilibrium(T::Float32, ux::Float32, uy::Float32, uz::Float32,
                                                w_k::Float32, cx::Float32, cy::Float32, cz::Float32)
    cu = cx*ux + cy*uy + cz*uz
    return w_k * T * (1.0f0 + 3.0f0*cu)
end


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