"""
DIAGNOSTICS.JL - Diagnostic computations for LBM solver
"""

using KernelAbstractions
using CUDA
using Printf

"""
Compute vorticity magnitude at each cell.
"""
@kernel function vorticity_kernel!(w_mag, vel, nx, ny, nz)
    i, j, k, b = @index(Global, NTuple)
    
    if i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
        @inbounds begin
            duz_dy = 0.5f0 * (vel[i, j+1, k, b, 3] - vel[i, j-1, k, b, 3])
            duy_dz = 0.5f0 * (vel[i, j, k+1, b, 2] - vel[i, j, k-1, b, 2])
            wx = duz_dy - duy_dz

            dux_dz = 0.5f0 * (vel[i, j, k+1, b, 1] - vel[i, j, k-1, b, 1])
            duz_dx = 0.5f0 * (vel[i+1, j, k, b, 3] - vel[i-1, j, k, b, 3])
            wy = dux_dz - duz_dx

            duy_dx = 0.5f0 * (vel[i+1, j, k, b, 2] - vel[i-1, j, k, b, 2])
            dux_dy = 0.5f0 * (vel[i, j+1, k, b, 1] - vel[i, j-1, k, b, 1])
            wz = duy_dx - dux_dy

            w_mag[i, j, k, b] = sqrt(wx*wx + wy*wy + wz*wz)
        end
    else
        @inbounds w_mag[i, j, k, b] = 0.0f0
    end
end

"""
Compute vorticity magnitude field for a BlockLevel.
"""
function compute_vorticity(level::BlockLevel)
    backend = get_backend(level.vel)
    n_blocks = length(level.active_block_coords)
    w_mag = KernelAbstractions.zeros(backend, Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    
    if n_blocks > 0
        kernel! = vorticity_kernel!(backend)
        kernel!(w_mag, level.vel, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 
                ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks))
    end
    
    return w_mag
end

"""
Compute basic flow statistics.
"""
function compute_flow_stats(level::BlockLevel)
    backend = get_backend(level.rho)
    
    if backend isa CUDABackend
        rho_flat = reshape(level.rho, :)
        vel_flat = reshape(level.vel, :, 3)
        obs_flat = reshape(level.obstacle, :)
        
        valid_mask = .!obs_flat
        n_fluid = sum(valid_mask)
        
        if n_fluid > 0
            # Density stats
            rho_fluid = rho_flat .* valid_mask
            rho_mean = sum(rho_fluid) / n_fluid
            rho_min = minimum(rho_flat[obs_flat .== false])
            rho_max = maximum(rho_flat[obs_flat .== false])
            
            # Velocity stats
            v2 = sum(abs2, vel_flat, dims=2)
            v_mag = sqrt.(vec(v2))
            v_max = maximum(v_mag .* valid_mask)
            
            # Kinetic energy
            ke = 0.5f0 * sum(rho_flat .* vec(v2) .* valid_mask)
            
            return (
                n_fluid = n_fluid,
                rho_mean = rho_mean,
                rho_min = rho_min,
                rho_max = rho_max,
                v_max = v_max,
                kinetic_energy = ke
            )
        end
    end
    
    return (n_fluid=0, rho_mean=1.0, rho_min=1.0, rho_max=1.0, v_max=0.0, kinetic_energy=0.0)
end

"""
Check simulation stability and print warnings.
"""
function check_stability(level::BlockLevel, step::Int)
    stats = compute_flow_stats(level)
    
    warnings = String[]
    
    if stats.v_max > 0.3
        push!(warnings, @sprintf("High velocity: %.4f (Ma > 0.5)", stats.v_max))
    end
    
    if stats.rho_min < 0.5
        push!(warnings, @sprintf("Low density: %.4f", stats.rho_min))
    end
    
    if stats.rho_max > 1.5
        push!(warnings, @sprintf("High density: %.4f", stats.rho_max))
    end
    
    if !isempty(warnings)
        println("[WARNING] Step $step stability issues:")
        for w in warnings
            println("  - $w")
        end
        return false
    end
    
    return true
end
