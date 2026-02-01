// # FILE: .\src\io_vtk.jl
using WriteVTK
using Printf
using Base.Threads

"""
IO_VTK.JL - VTK Output Logic

Handles the aggregation of multi-level block data and writes structured
VTK XML (vtu/vtm) files for visualization in ParaView.
"""

function export_merged_mesh_sync(t_step, grids, out_dir, backend)
    log_walltime("VTK Export START for step $t_step")
    vtk_start = time()
    
    # 1. Identify valid blocks (exclude those covered by finer levels)
    level_blocks = [Set{Tuple{Int,Int,Int}}() for _ in 1:length(grids)]
    for lvl in 1:length(grids)
        for b in grids[lvl].active_block_coords
            push!(level_blocks[lvl], b)
        end
    end
    
    valid_blocks = Vector{NamedTuple{(:lvl, :b_idx, :bx, :by, :bz), Tuple{Int, Int, Int, Int, Int}}}()
    
    for lvl in 1:length(grids)
        next_blocks = lvl < length(grids) ? level_blocks[lvl+1] : nothing
        for (b_idx, (bx, by, bz)) in enumerate(grids[lvl].active_block_coords)
            should_export = true
            if next_blocks !== nothing
                children = 0
                for dbz in 0:1, dby in 0:1, dbx in 0:1
                    if (2*bx-1+dbx, 2*by-1+dby, 2*bz-1+dbz) in next_blocks
                        children += 1
                    end
                end
                # If all 8 children exist, this block is fully refined. Don't export.
                if children == 8
                    should_export = false
                end
            end
            if should_export
                push!(valid_blocks, (lvl=lvl, b_idx=b_idx, bx=bx, by=by, bz=bz))
            end
        end
    end
    
    if isempty(valid_blocks); return; end
    
    # 2. Gather data from GPU to CPU
    level_data = Dict{Int, NamedTuple}()
    for lvl in 1:length(grids)
        level = grids[lvl]
        rho_cpu = Array(level.rho)
        vel_cpu = iseven(t_step) ? Array(level.vel_temp) : Array(level.vel)
        obs_cpu = Array(level.obstacle)
        level_data[lvl] = (rho=rho_cpu, vel=vel_cpu, obs=obs_cpu, dx=Float32(level.dx))
    end
    
    n_total = length(valid_blocks)
    n_pts = (BLOCK_SIZE + 1)^3
    n_cells = BLOCK_SIZE^3
    
    points = Matrix{Float32}(undef, 3, n_total * n_pts)
    cells = Vector{MeshCell}(undef, n_total * n_cells)
    rho_arr = Vector{Float32}(undef, n_total * n_cells)
    vel_mat = Matrix{Float32}(undef, 3, n_total * n_cells)
    obst_arr = Vector{UInt8}(undef, n_total * n_cells)
    level_arr = Vector{Int32}(undef, n_total * n_cells)
    
    # 3. Construct unstructured grid
    Threads.@threads for i in 1:n_total
        blk = valid_blocks[i]
        data = level_data[blk.lvl]
        dx = data.dx
        
        pt_off = (i - 1) * n_pts
        cell_off = (i - 1) * n_cells
        off_x = Float32((blk.bx - 1) * BLOCK_SIZE)
        off_y = Float32((blk.by - 1) * BLOCK_SIZE)
        off_z = Float32((blk.bz - 1) * BLOCK_SIZE)
        
        # Generate Points
        for pz in 0:BLOCK_SIZE, py in 0:BLOCK_SIZE, px in 0:BLOCK_SIZE
            pidx = pt_off + 1 + px + py*(BLOCK_SIZE+1) + pz*(BLOCK_SIZE+1)^2
            points[1, pidx] = (off_x + px) * dx
            points[2, pidx] = (off_y + py) * dx
            points[3, pidx] = (off_z + pz) * dx
        end
        
        stride_y = BLOCK_SIZE + 1
        stride_z = (BLOCK_SIZE + 1)^2
        
        # Generate Cells and Fields
        for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
            cidx = cell_off + x + (y-1)*BLOCK_SIZE + (z-1)*BLOCK_SIZE^2
            pt_base = pt_off + 1 + (x-1) + (y-1)*stride_y + (z-1)*stride_z
            cells[cidx] = MeshCell(VTKCellTypes.VTK_VOXEL,
                (pt_base, pt_base+1, pt_base+stride_y, pt_base+stride_y+1,
                 pt_base+stride_z, pt_base+stride_z+1, pt_base+stride_z+stride_y, pt_base+stride_z+stride_y+1))
            
            vel_mat[1, cidx] = data.vel[x, y, z, blk.b_idx, 1]
            vel_mat[2, cidx] = data.vel[x, y, z, blk.b_idx, 2]
            vel_mat[3, cidx] = data.vel[x, y, z, blk.b_idx, 3]
            rho_arr[cidx] = data.rho[x, y, z, blk.b_idx]
            obst_arr[cidx] = data.obs[x, y, z, blk.b_idx] ? 0x01 : 0x00
            level_arr[cidx] = Int32(blk.lvl)
        end
    end
    
    replace!(rho_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    replace!(vel_mat, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    
    filename = @sprintf("%s/flow_%06d", out_dir, t_step)
    try
        vtk_grid(filename, points, cells; compress=true, append=false) do vtk
            if OUTPUT_DENSITY; vtk["Density"] = rho_arr; end
            if OUTPUT_VELOCITY; vtk["Velocity"] = vel_mat; end
            if OUTPUT_VEL_MAG; vtk["VelocityMagnitude"] = sqrt.(vel_mat[1,:].^2 .+ vel_mat[2,:].^2 .+ vel_mat[3,:].^2); end
            if OUTPUT_OBSTACLE; vtk["Obstacle"] = obst_arr; end
            if OUTPUT_LEVEL; vtk["Level"] = level_arr; end
        end
    catch e
        println("[Error] VTK write failed: $e")
    end
    
    log_walltime(@sprintf("VTK Export END (%.2fM cells, %.1fs)", n_total*n_cells/1e6, time()-vtk_start))
end