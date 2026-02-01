# FILE: src/domain_generation.jl
using Base.Threads
using LinearAlgebra
using StaticArrays

"""
DOMAIN_GENERATION.JL - Voxelization, Sponge Layers, and Geometry Processing
"""

function triangle_intersects_aabb(center, box_half_size, v1, v2, v3)
    tol = 1.001
    h = box_half_size * tol
    t1 = v1 - center; t2 = v2 - center; t3 = v3 - center
    
    if min(t1[1], t2[1], t3[1]) > h[1] || max(t1[1], t2[1], t3[1]) < -h[1]; return false; end
    if min(t1[2], t2[2], t3[2]) > h[2] || max(t1[2], t2[2], t3[2]) < -h[2]; return false; end
    if min(t1[3], t2[3], t3[3]) > h[3] || max(t1[3], t2[3], t3[3]) < -h[3]; return false; end
    
    f = [t2 - t1, t3 - t2, t1 - t3]
    u = [SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0)]
    
    for i in 1:3
        for j in 1:3
            axis = cross(u[i], f[j])
            if dot(axis, axis) < 1e-10; continue; end
            p1 = dot(t1, axis); p2 = dot(t2, axis); p3 = dot(t3, axis)
            r = h[1]*abs(axis[1]) + h[2]*abs(axis[2]) + h[3]*abs(axis[3])
            if min(p1, min(p2, p3)) > r || max(p1, max(p2, p3)) < -r; return false; end
        end
    end
    return true
end

function build_block_triangle_map(mesh::Geometry.SolverMesh, 
                                  sorted_blocks::Vector{Tuple{Int, Int, Int}}, 
                                  dx::Float64,
                                  mesh_offset::SVector{3,Float64})
    block_tris = [Int[] for _ in 1:length(sorted_blocks)]
    bs = BLOCK_SIZE
    
    b_lookup = Dict{Tuple{Int, Int, Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        b_lookup[coord] = i
    end
    
    margin = dx * 2
    
    for (t_idx, tri) in enumerate(mesh.triangles)
        min_pt = tri[1] .+ mesh_offset
        max_pt = tri[1] .+ mesh_offset
        for p in tri
             pm = p .+ mesh_offset
             min_pt = min.(min_pt, pm)
             max_pt = max.(max_pt, pm)
        end
        
        min_b = floor.(Int, (min_pt .- margin) ./ (bs * dx)) .+ 1
        max_b = floor.(Int, (max_pt .+ margin) ./ (bs * dx)) .+ 1
        
        for bz in max(1, min_b[3]):max_b[3]
            for by in max(1, min_b[2]):max_b[2]
                for bx in max(1, min_b[1]):max_b[1]
                    if haskey(b_lookup, (bx, by, bz))
                        idx = b_lookup[(bx, by, bz)]
                        push!(block_tris[idx], t_idx)
                    end
                end
            end
        end
    end
    return block_tris
end

function voxelize_blocks!(obstacle_arr, sorted_blocks, mesh::Geometry.SolverMesh, 
                          dx::Float64, mesh_offset::SVector{3,Float64})
    block_tri_map = build_block_triangle_map(mesh, sorted_blocks, dx, mesh_offset)
    
    println("[Domain] SAT Surface Marking...")
    box_half = SVector(0.75, 0.75, 0.75) * dx
    
    @threads for i in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[i]
        relevant_tris = block_tri_map[i]
        if isempty(relevant_tris)
            continue
        end
        
        for lz in 1:BLOCK_SIZE, ly in 1:BLOCK_SIZE, lx in 1:BLOCK_SIZE
            px = ((bx - 1) * BLOCK_SIZE + lx - 0.5) * dx
            py = ((by - 1) * BLOCK_SIZE + ly - 0.5) * dx
            pz = ((bz - 1) * BLOCK_SIZE + lz - 0.5) * dx
            center = SVector(px, py, pz)
            
            is_shell = false
            for tid in relevant_tris
                t = mesh.triangles[tid]
                v1 = t[1] .+ mesh_offset
                v2 = t[2] .+ mesh_offset
                v3 = t[3] .+ mesh_offset
                
                if triangle_intersects_aabb(center, box_half, v1, v2, v3)
                    is_shell = true
                    break
                end
            end
            
            if is_shell
                obstacle_arr[lx, ly, lz, i] = true
            end
        end
    end
end

function perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, 
                             grid_dim_x, grid_dim_y, grid_dim_z)
    println("[Domain] Flood Fill...")
    visited = falses(size(obstacle_arr))
    q_block = Int32[]
    q_idx = Int32[]
    
    min_x_block = minimum(b[1] for b in sorted_blocks)
    
    for b_idx in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[b_idx]
        if bx == min_x_block
            for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
                if !obstacle_arr[x, y, z, b_idx]
                    visited[x, y, z, b_idx] = true
                    push!(q_block, b_idx)
                    push!(q_idx, x + (y-1)*BLOCK_SIZE + (z-1)*BLOCK_SIZE*BLOCK_SIZE)
                end
            end
        end
    end
    
    head = 1
    dx_arr = [1, -1, 0, 0, 0, 0]
    dy_arr = [0, 0, 1, -1, 0, 0]
    dz_arr = [0, 0, 0, 0, 1, -1]
    bs = BLOCK_SIZE
    bs2 = bs*bs
    
    function has_block_local(bx, by, bz)
        if bx < 1 || bx > grid_dim_x || by < 1 || by > grid_dim_y || bz < 1 || bz > grid_dim_z
            return false
        end
        return block_ptr[bx, by, bz] > 0
    end
    
    while head <= length(q_block)
        b_curr = q_block[head]
        idx_curr = q_idx[head]
        head += 1
        
        rem = idx_curr - 1
        lz = rem ÷ bs2 + 1
        rem = rem % bs2
        ly = rem ÷ bs + 1
        lx = rem % bs + 1
        
        (bx, by, bz) = sorted_blocks[b_curr]
        
        for i in 1:6
            nlx = lx + dx_arr[i]
            nly = ly + dy_arr[i]
            nlz = lz + dz_arr[i]
            
            if nlx >= 1 && nlx <= bs && nly >= 1 && nly <= bs && nlz >= 1 && nlz <= bs
                if !visited[nlx, nly, nlz, b_curr] && !obstacle_arr[nlx, nly, nlz, b_curr]
                    visited[nlx, nly, nlz, b_curr] = true
                    push!(q_block, b_curr)
                    push!(q_idx, nlx + (nly-1)*bs + (nlz-1)*bs2)
                end
            else
                nbx = bx + dx_arr[i]
                nby = by + dy_arr[i]
                nbz = bz + dz_arr[i]
                
                if has_block_local(nbx, nby, nbz)
                    nb_idx = block_ptr[nbx, nby, nbz]
                    wlx = (nlx < 1) ? bs : (nlx > bs ? 1 : nlx)
                    wly = (nly < 1) ? bs : (nly > bs ? 1 : nly)
                    wlz = (nlz < 1) ? bs : (nlz > bs ? 1 : nlz)
                    
                    if !visited[wlx, wly, wlz, nb_idx] && !obstacle_arr[wlx, wly, wlz, nb_idx]
                        visited[wlx, wly, wlz, nb_idx] = true
                        push!(q_block, nb_idx)
                        push!(q_idx, wlx + (wly-1)*bs + (wlz-1)*bs2)
                    end
                end
            end
        end
    end
    
    filled_count = 0
    for i in 1:length(obstacle_arr)
        if !obstacle_arr[i] && !visited[i]
            obstacle_arr[i] = true
            filled_count += 1
        end
    end
    println("[Domain] Filled $filled_count interior voxels.")
end

@inline function smooth_sponge_profile(x::Float64, thickness::Float64)
    if x <= 0.0
        return 1.0
    elseif x >= thickness
        return 0.0
    else
        return 0.5 * (1.0 + cos(π * x / thickness))
    end
end

function apply_sponge!(sponge_arr, sorted_blocks, params, lvl_scale::Int)
    println("[Domain] Applying Sponge Layers...")
    dx = params.dx_coarse / lvl_scale
    
    Lx = params.domain_size[1]
    Ly = params.domain_size[2]
    Lz = params.domain_size[3]
    
    outlet_thickness = Lx * max(Float64(SPONGE_THICKNESS), 0.15) 
    
    inlet_thickness = Lx * 0.02  
    y_sponge_thickness = Ly * Float64(SPONGE_THICKNESS) * 0.5
    z_sponge_thickness = Lz * Float64(SPONGE_THICKNESS) * 0.5
    
    outlet_start = Lx - outlet_thickness
    y_top_start = Ly - y_sponge_thickness
    z_back_start = Lz - z_sponge_thickness
    
    outlet_strength = 1.0      
    inlet_strength = 0.05      
    wall_strength = 0.1        
    
    for i in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[i]
        
        for lz in 1:BLOCK_SIZE, ly in 1:BLOCK_SIZE, lx in 1:BLOCK_SIZE
            px = ((bx - 1) * BLOCK_SIZE + lx - 0.5) * dx
            py = ((by - 1) * BLOCK_SIZE + ly - 0.5) * dx
            pz = ((bz - 1) * BLOCK_SIZE + lz - 0.5) * dx
            
            sponge_val = 0.0
            
            if px > outlet_start
                dist_from_boundary = px - outlet_start
                s = smooth_sponge_profile(outlet_thickness - dist_from_boundary, outlet_thickness)
                sponge_val = max(sponge_val, s * outlet_strength)
            end
            
            if px < inlet_thickness
                s = smooth_sponge_profile(px, inlet_thickness)
                sponge_val = max(sponge_val, s * inlet_strength)
            end
            
            if !SYMMETRIC_ANALYSIS && py < y_sponge_thickness
                s = smooth_sponge_profile(py, y_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            if py > y_top_start
                dist_from_boundary = py - y_top_start
                s = smooth_sponge_profile(y_sponge_thickness - dist_from_boundary, y_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            if pz < z_sponge_thickness
                s = smooth_sponge_profile(pz, z_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            if pz > z_back_start
                dist_from_boundary = pz - z_back_start
                s = smooth_sponge_profile(z_sponge_thickness - dist_from_boundary, z_sponge_thickness)
                sponge_val = max(sponge_val, s * wall_strength)
            end
            
            sponge_arr[lx, ly, lz, i] = Float32(sponge_val)
        end
    end
    
    sponge_cells = count(sponge_arr .> 0.0f0)
    total_cells = length(sponge_arr)
    max_sponge = maximum(sponge_arr)
    @printf("[Domain] Sponge: %.1f%% cells affected, max strength = %.3f\n", 
            100.0 * sponge_cells / total_cells, max_sponge)
end

function point_triangle_distance_sq(P_t, A_t, B_t, C_t)
    P = SVector(P_t)
    A = SVector(A_t)
    B = SVector(B_t)
    C = SVector(C_t)

    e0 = B - A
    e1 = C - A
    v0 = A - P

    a = dot(e0, e0)
    b = dot(e0, e1)
    c = dot(e1, e1)
    d = dot(e0, v0)
    e = dot(e1, v0)

    det = a*c - b*b
    s = b*e - c*d
    t = b*d - a*e

    if (s + t <= det)
        if (s < 0.0)
            if (t < 0.0)  
                if (d < 0.0)
                    t = 0.0
                    s = (-d >= a ? 1.0 : -d/a)
                else
                    s = 0.0
                    t = (e >= 0.0 ? 0.0 : (-e >= c ? 1.0 : -e/c))
                end
            else  
                s = 0.0
                t = (e >= 0.0 ? 0.0 : (-e >= c ? 1.0 : -e/c))
            end
        elseif (t < 0.0)  
            t = 0.0
            s = (d >= 0.0 ? 0.0 : (-d >= a ? 1.0 : -d/a))
        else  
            invDet = 1.0 / det
            s *= invDet
            t *= invDet
        end
    else
        if (s < 0.0)  
            tmp0 = b + d
            tmp1 = c + e
            if (tmp1 > tmp0)
                numer = tmp1 - tmp0
                denom = a - 2*b + c
                s = (numer >= denom ? 1.0 : numer/denom)
                t = 1.0 - s
            else
                s = 0.0
                t = (tmp1 <= 0.0 ? 1.0 : (e >= 0.0 ? 0.0 : -e/c))
            end
        elseif (t < 0.0)  
            tmp0 = b + e
            tmp1 = a + d
            if (tmp1 > tmp0)
                numer = tmp1 - tmp0
                denom = a - 2*b + c
                t = (numer >= denom ? 1.0 : numer/denom)
                s = 1.0 - t
            else
                t = 0.0
                s = (tmp1 <= 0.0 ? 1.0 : (d >= 0.0 ? 0.0 : -d/a))
            end
        else  
            numer = c + e - b - d
            denom = a - 2*b + c
            s = (numer >= denom ? 1.0 : numer/denom)
            t = 1.0 - s
        end
    end

    Q = A + s*e0 + t*e1
    diff = P - Q
    return dot(diff, diff)
end

function compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, 
                                 mesh::Geometry.SolverMesh, dx::Float64, 
                                 mesh_offset::SVector{3,Float64})
    println("[Domain] Computing wall distances...")
    bs = BLOCK_SIZE
    n_blocks = length(sorted_blocks)
    
    block_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        block_lookup[coord] = i
    end
    
    near_wall_count = 0
    
    @threads for b_idx in 1:n_blocks
        (bx, by, bz) = sorted_blocks[b_idx]
        
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            if obstacle_arr[lx, ly, lz, b_idx]
                continue  
            end
            
            is_near_wall = false
            min_dist = 100.0f0
            
            for dz in -1:1, dy in -1:1, dx_off in -1:1
                if dx_off == 0 && dy == 0 && dz == 0
                    continue
                end
                
                nx, ny, nz = lx + dx_off, ly + dy, lz + dz
                nb_idx = b_idx
                
                if nx < 1 || nx > bs || ny < 1 || ny > bs || nz < 1 || nz > bs
                    nbx = bx + (nx < 1 ? -1 : (nx > bs ? 1 : 0))
                    nby = by + (ny < 1 ? -1 : (ny > bs ? 1 : 0))
                    nbz = bz + (nz < 1 ? -1 : (nz > bs ? 1 : 0))
                    
                    if haskey(block_lookup, (nbx, nby, nbz))
                        nb_idx = block_lookup[(nbx, nby, nbz)]
                        nx = nx < 1 ? nx + bs : (nx > bs ? nx - bs : nx)
                        ny = ny < 1 ? ny + bs : (ny > bs ? ny - bs : ny)
                        nz = nz < 1 ? nz + bs : (nz > bs ? nz - bs : nz)
                    else
                        continue
                    end
                end
                
                if obstacle_arr[nx, ny, nz, nb_idx]
                    is_near_wall = true
                    dist = sqrt(Float32(dx_off^2 + dy^2 + dz^2)) * Float32(dx)
                    min_dist = min(min_dist, dist)
                end
            end
            
            if is_near_wall
                wall_dist_arr[lx, ly, lz, b_idx] = min_dist
                near_wall_count += 1
            end
        end
    end
    
    println("[Domain] Found $near_wall_count near-wall cells")
end