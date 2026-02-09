using KernelAbstractions
using CUDA
using Adapt


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

    # Thermal DDF ping-pong (same parity as flow)
    local g_in_t, g_out_t
    ddf_active = DDF_ENABLED && has_thermal(level)
    if ddf_active
        if iseven(t_sub)
            g_in_t, g_out_t = level.g, level.g_temp
        else
            g_in_t, g_out_t = level.g_temp, level.g
        end
    end

    has_children = current_lvl < length(grids)

    if has_children && use_temporal_interp && has_temporal_storage(level)
        copy_to_old!(level, f_in, vel_in; g_current = ddf_active ? g_in_t : nothing)
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
                         t_sub, inlet_turbulence, 0.0f0, use_temporal_interp, sponge_blend_dist;
                         g_out = ddf_active ? g_out_t : nothing,
                         g_in = ddf_active ? g_in_t : nothing,
                         g_post_collision_arr = ddf_active ? level.g_post_collision : nothing,
                         parent_g = nothing,
                         parent_temperature = nothing,
                         parent_g_old = nothing,
                         parent_temperature_old = nothing,
                         parent_vel_thermal = nothing,
                         parent_vel_old_thermal = nothing,
                         parent_tau_g = 0.5f0)

    if has_children
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.0f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0)

        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub + 1,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.5f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0)
    end
end

function recursive_step_temporal!(grids, current_lvl::Int, t_sub::Int,
                                  parent_f, parent_rho, parent_vel, parent_ptr,
                                  parent_f_old, parent_rho_old, parent_vel_old,
                                  parent_tau::Float32, temporal_weight::Float32, u_vel::Float32,
                                  cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                  domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                  wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                  inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool;
                                  # Thermal DDF parent data (Phase 2)
                                  parent_g_data=nothing, parent_temperature_data=nothing,
                                  parent_g_old_data=nothing, parent_temperature_old_data=nothing,
                                  parent_vel_for_thermal=nothing, parent_vel_old_for_thermal=nothing,
                                  parent_tau_g_val::Float32=0.5f0)

    if current_lvl > length(grids); return; end

    level = grids[current_lvl]

    if iseven(t_sub)
        f_in, f_out = level.f, level.f_temp
        vel_in, vel_out = level.vel, level.vel_temp
    else
        f_in, f_out = level.f_temp, level.f
        vel_in, vel_out = level.vel_temp, level.vel
    end

    # Thermal DDF ping-pong
    local g_in_t, g_out_t
    ddf_active = DDF_ENABLED && has_thermal(level)
    if ddf_active
        if iseven(t_sub)
            g_in_t, g_out_t = level.g, level.g_temp
        else
            g_in_t, g_out_t = level.g_temp, level.g
        end
    end

    has_children = current_lvl < length(grids)

    if has_children && use_temporal_interp && has_temporal_storage(level)
        copy_to_old!(level, f_in, vel_in; g_current = ddf_active ? g_in_t : nothing)
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
                         t_sub, inlet_turbulence, temporal_weight, use_temporal_interp, sponge_blend_dist;
                         g_out = ddf_active ? g_out_t : nothing,
                         g_in = ddf_active ? g_in_t : nothing,
                         g_post_collision_arr = ddf_active ? level.g_post_collision : nothing,
                         parent_g = parent_g_data,
                         parent_temperature = parent_temperature_data,
                         parent_g_old = parent_g_old_data,
                         parent_temperature_old = parent_temperature_old_data,
                         parent_vel_thermal = parent_vel_for_thermal,
                         parent_vel_old_thermal = parent_vel_old_for_thermal,
                         parent_tau_g = parent_tau_g_val)

    if has_children
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.0f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0)

        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub + 1,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, 0.5f0, u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0)
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