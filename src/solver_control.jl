// # FILE: .\src\solver_control.jl
using KernelAbstractions
using CUDA
using Adapt

"""
SOLVER_CONTROL.JL - Time Stepping Logic

Contains the recursive multi-grid stepping algorithm including
temporal interpolation for fine grid boundary conditions.
"""

"""
Recursive timestep for multi-level grids with temporal interpolation.

Before stepping coarse level: save state to "old" arrays.
Fine level sees smoothly varying boundary conditions:
  - Sub-step 0: temporal_weight = 0.0 (use 100% old coarse data)
  - Sub-step 1: temporal_weight = 0.5 (blend 50% old + 50% new)
"""
function recursive_step!(grids, current_lvl::Int, t_sub::Int,
                         parent_f, parent_rho, parent_vel, parent_ptr,
                         parent_f_old, parent_rho_old, parent_vel_old,
                         parent_tau::Float32, u_vel::Float32,
                         cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                         domain_nx::Int, domain_ny::Int, domain_nz::Int,
                         wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                         inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)
    
    if current_lvl > length(grids); return; end
    
    level = grids[current_lvl]
    
    
    if iseven(t_sub)
        f_in, f_out = level.f, level.f_temp
        vel_in, vel_out = level.vel, level.vel_temp
    else
        f_in, f_out = level.f_temp, level.f
        vel_in, vel_out = level.vel_temp, level.vel
    end
    
    has_children = current_lvl < length(grids)
    
    # Save state BEFORE stepping for children's temporal interpolation
    if has_children && use_temporal_interp && has_temporal_storage(level)
        copy_to_old!(level, f_in, vel_in)
    end
    
    
    perform_timestep_v2!(level,
                         parent_f, parent_rho, parent_vel, parent_ptr,
                         parent_f_old, parent_rho_old, parent_vel_old,
                         parent_tau,
                         f_out, f_in, vel_out, vel_in,
                         level.f_post_collision, u_vel,
                         cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                         domain_nx, domain_ny, domain_nz,
                         wall_model_active, c_wale_val, nu_sgs_bg,
                         t_sub, inlet_turbulence, 0.0f0, use_temporal_interp, sponge_blend_dist)
    
    
    if has_children
        
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.0f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist)
        
        
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub + 1,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.5f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist)
    end
end

function recursive_step_temporal!(grids, current_lvl::Int, t_sub::Int,
                                  parent_f, parent_rho, parent_vel, parent_ptr,
                                  parent_f_old, parent_rho_old, parent_vel_old,
                                  parent_tau::Float32, temporal_weight::Float32, u_vel::Float32,
                                  cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                  domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                  wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                  inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)
    
    if current_lvl > length(grids); return; end
    
    level = grids[current_lvl]
    
    if iseven(t_sub)
        f_in, f_out = level.f, level.f_temp
        vel_in, vel_out = level.vel, level.vel_temp
    else
        f_in, f_out = level.f_temp, level.f
        vel_in, vel_out = level.vel_temp, level.vel
    end
    
    has_children = current_lvl < length(grids)
    
    if has_children && use_temporal_interp && has_temporal_storage(level)
        copy_to_old!(level, f_in, vel_in)
    end
    
    perform_timestep_v2!(level,
                         parent_f, parent_rho, parent_vel, parent_ptr,
                         parent_f_old, parent_rho_old, parent_vel_old,
                         parent_tau,
                         f_out, f_in, vel_out, vel_in,
                         level.f_post_collision, u_vel,
                         cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                         domain_nx, domain_ny, domain_nz,
                         wall_model_active, c_wale_val, nu_sgs_bg,
                         t_sub, inlet_turbulence, temporal_weight, use_temporal_interp, sponge_blend_dist)
    
    if has_children
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.0f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist)
        
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub + 1,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.5f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist)
    end
end

function execute_timestep_batch!(grids, t_start::Int, batch_size::Int, u_curr::Float32,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                 wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                 inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)
    backend = get_backend(grids[1].rho)
    
    for t_offset in 0:(batch_size-1)
        t = t_start + t_offset
        recursive_step!(grids, 1, t,
                        nothing, nothing, nothing, nothing,
                        nothing, nothing, nothing,
                        Float32(0.5), u_curr,
                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                        domain_nx, domain_ny, domain_nz,
                        wall_model_active, c_wale_val, nu_sgs_bg,
                        inlet_turbulence, use_temporal_interp, sponge_blend_dist)
    end
    
    KernelAbstractions.synchronize(backend)
end