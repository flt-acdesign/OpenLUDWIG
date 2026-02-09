# GLOBAL.JL - Global Aerodynamic Force Computation
#
# Computes total forces and moments using the momentum exchange method.
# This sums up all boundary link contributions from Bouzidi cells.

using KernelAbstractions
using Atomix

"""
GPU kernel to compute global forces and moments via momentum exchange method.

Each Bouzidi boundary cell contributes momentum exchange from all intersecting links.
Forces are computed as F = 2 * sum(f_k * c_k) for all solid-hitting directions.
"""
@kernel function global_mem_kernel!(Fx, Fy, Fz, Mx, My, Mz,
                                    f_post, q_map, 
                                    cell_block, cell_x, cell_y, cell_z,
                                    map_x, map_y, map_z,
                                    cx_arr, cy_arr, cz_arr,
                                    block_size::Int32, dx::Float32,
                                    mc_x::Float32, mc_y::Float32, mc_z::Float32)
    idx = @index(Global)
    @inbounds begin
        b_idx = cell_block[idx]
        lx = cell_x[idx]; ly = cell_y[idx]; lz = cell_z[idx]
        
        # Compute physical position of this cell
        bx = map_x[b_idx]; by = map_y[b_idx]; bz = map_z[b_idx]
        px = ((bx - 1) * block_size + lx - 0.5f0) * dx
        py = ((by - 1) * block_size + ly - 0.5f0) * dx
        pz = ((bz - 1) * block_size + lz - 0.5f0) * dx
        
        # Moment arm from moment center
        rx = px - mc_x
        ry = py - mc_y
        rz = pz - mc_z
        
        local_fx = 0.0f0
        local_fy = 0.0f0
        local_fz = 0.0f0
        
        # Sum momentum exchange from all boundary links
        for k in 1:27
            q = Float32(q_map[lx, ly, lz, b_idx, k])
            if q > 0.0f0 && q <= 1.0f0
                # Distribution hitting the wall
                val = f_post[lx, ly, lz, b_idx, k]
                
                # Momentum exchange: 2 * f * c (factor of 2 from bounce-back)
                kick = 2.0f0 * val
                
                local_fx += kick * Float32(cx_arr[k])
                local_fy += kick * Float32(cy_arr[k])
                local_fz += kick * Float32(cz_arr[k])
            end
        end
        
        # Atomic accumulation of forces
        Atomix.@atomic Fx[1] += local_fx
        Atomix.@atomic Fy[1] += local_fy
        Atomix.@atomic Fz[1] += local_fz
        
        # Moments: M = r × F
        Atomix.@atomic Mx[1] += (ry * local_fz - rz * local_fy)
        Atomix.@atomic My[1] += (rz * local_fx - rx * local_fz)
        Atomix.@atomic Mz[1] += (rx * local_fy - ry * local_fx)
    end
end

"""
    compute_global_aerodynamics!(force_data, level, f_post, cx_gpu, cy_gpu, cz_gpu, backend)

Compute total aerodynamic forces and moments using the momentum exchange method.
Updates force_data in place with dimensional forces/moments and coefficients.
"""
function compute_global_aerodynamics!(force_data::ForceData, level, f_post,
                                      cx_gpu, cy_gpu, cz_gpu, backend)
    if !level.bouzidi_enabled || level.n_boundary_cells == 0
        return
    end
    
    # Allocate accumulators on GPU
    d_Fx = KernelAbstractions.zeros(backend, Float32, 1)
    d_Fy = KernelAbstractions.zeros(backend, Float32, 1)
    d_Fz = KernelAbstractions.zeros(backend, Float32, 1)
    d_Mx = KernelAbstractions.zeros(backend, Float32, 1)
    d_My = KernelAbstractions.zeros(backend, Float32, 1)
    d_Mz = KernelAbstractions.zeros(backend, Float32, 1)
    
    mc = force_data.moment_center
    
    kernel! = global_mem_kernel!(backend)
    kernel!(d_Fx, d_Fy, d_Fz, d_Mx, d_My, d_Mz,
            f_post, level.bouzidi_q_map,
            level.bouzidi_cell_block, level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
            level.map_x, level.map_y, level.map_z,
            cx_gpu, cy_gpu, cz_gpu,
            Int32(BLOCK_SIZE), Float32(level.dx),
            Float32(mc[1]), Float32(mc[2]), Float32(mc[3]),
            ndrange=(level.n_boundary_cells,))
    KernelAbstractions.synchronize(backend)
    
    # Convert from lattice to physical units
    fs = force_data.force_scale
    ls = force_data.length_scale
    
    Fx_lat = Array(d_Fx)[1]
    Fy_lat = Array(d_Fy)[1]
    Fz_lat = Array(d_Fz)[1]
    Mx_lat = Array(d_Mx)[1]
    My_lat = Array(d_My)[1]
    Mz_lat = Array(d_Mz)[1]
    
    # Handle symmetry: double X and Z forces, Y force and roll/yaw moments are zero
    if force_data.symmetric
        Fx_lat *= 2.0
        Fz_lat *= 2.0
        My_lat *= 2.0
        Fy_lat = 0.0
        Mx_lat = 0.0
        Mz_lat = 0.0
    end
    
    # Store dimensional forces and moments
    force_data.Fx = Fx_lat * fs
    force_data.Fy = Fy_lat * fs
    force_data.Fz = Fz_lat * fs
    force_data.Mx = Mx_lat * fs * ls
    force_data.My = My_lat * fs * ls
    force_data.Mz = Mz_lat * fs * ls
    
    # Compute non-dimensional coefficients
    q_dyn = 0.5 * force_data.rho_ref * force_data.u_ref^2
    F_ref = q_dyn * force_data.area_ref
    M_ref = F_ref * force_data.chord_ref
    
    if F_ref > 1e-10
        force_data.Cd = force_data.Fx / F_ref   # Drag coefficient (X direction)
        force_data.Cl = force_data.Fz / F_ref   # Lift coefficient (Z direction) 
        force_data.Cs = force_data.Fy / F_ref   # Side force coefficient (Y direction)
    end
    
    if M_ref > 1e-10
        force_data.Cmx = force_data.Mx / M_ref  # Roll moment coefficient
        force_data.Cmy = force_data.My / M_ref  # Pitch moment coefficient
        force_data.Cmz = force_data.Mz / M_ref  # Yaw moment coefficient
    end
end
