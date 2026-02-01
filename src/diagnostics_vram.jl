# FILE: .\src\diagnostics_vram.jl
"""
DIAGNOSTICS_VRAM.JL - Memory Usage Analysis Tool (AA-PATTERN VERSION)

Provides detailed breakdown of VRAM consumption by component.
"""

using Printf
using CUDA

"""
    print_vram_breakdown(grids)

Print detailed VRAM usage breakdown by level and component.
Call after GPU transfer to see actual allocation.
"""
function print_vram_breakdown(grids)
    println("\n" * "="^70)
    println("                    VRAM BREAKDOWN BY LEVEL")
    println("="^70)
    
    total_dist = 0.0
    total_macro = 0.0
    total_bouzidi = 0.0
    total_geom = 0.0
    total_conn = 0.0
    
    for (lvl, g) in enumerate(grids)
        n_blocks = length(g.active_block_coords)
        n_cells = n_blocks * BLOCK_SIZE^3
        
        # Distribution functions (f, f_temp)
        # Note: We sum both because both exist in memory
        dist_f = sizeof(g.f) / 1024^2
        dist_temp = sizeof(g.f_temp) / 1024^2
        dist_total = dist_f + dist_temp
        
        # Macroscopic variables (rho, vel, vel_temp)
        macro_rho = sizeof(g.rho) / 1024^2
        macro_vel = sizeof(g.vel) / 1024^2
        macro_vt = sizeof(g.vel_temp) / 1024^2
        macro_total = macro_rho + macro_vel + macro_vt
        
        # Geometry arrays (obstacle, sponge, wall_dist)
        geom_obs = sizeof(g.obstacle) / 1024^2
        geom_sponge = sizeof(g.sponge) / 1024^2
        geom_wall = sizeof(g.wall_dist) / 1024^2
        geom_total = geom_obs + geom_sponge + geom_wall
        
        # Connectivity (pointers, neighbor tables)
        conn_nb = sizeof(g.neighbor_table) / 1024^2
        conn_ptr = sizeof(g.block_pointer) / 1024^2
        conn_maps = (sizeof(g.map_x) + sizeof(g.map_y) + sizeof(g.map_z)) / 1024^2
        conn_total = conn_nb + conn_ptr + conn_maps
        
        # Bouzidi data (q_map + sparse coords)
        if g.bouzidi_enabled
            bouz_qmap = sizeof(g.bouzidi_q_map) / 1024^2
            bouz_coords = (sizeof(g.bouzidi_cell_block) + sizeof(g.bouzidi_cell_x) + 
                           sizeof(g.bouzidi_cell_y) + sizeof(g.bouzidi_cell_z)) / 1024^2
            bouz_total = bouz_qmap + bouz_coords
        else
            bouz_qmap = 0.0
            bouz_coords = 0.0
            bouz_total = 0.0
        end
        
        level_total = dist_total + macro_total + geom_total + conn_total + bouz_total
        
        total_dist += dist_total
        total_macro += macro_total
        total_bouzidi += bouz_total
        total_geom += geom_total
        total_conn += conn_total
        
        # Print Level Summary
        @printf("\n┌─ Level %d: %d blocks, %.2fM cells ─────────────────────────\n", 
                lvl, n_blocks, n_cells/1e6)
        @printf("│  Distributions:      %7.1f MB  (f + f_temp)\n", dist_total)
        @printf("│  Macroscopic:        %7.1f MB  (rho, u, u_temp)\n", macro_total)
        @printf("│  Geometry:           %7.1f MB  (obs, sponge, wall)\n", geom_total)
        if g.bouzidi_enabled
            @printf("│  Bouzidi:            %7.1f MB  (%d boundary cells)\n", 
                    bouz_total, g.n_boundary_cells)
        end
        @printf("│  ────────────────────────────────────────────────────────\n")
        @printf("│  Level Total:        %7.1f MB\n", level_total)
        @printf("└──────────────────────────────────────────────────────────────\n")
    end
    
    # Grand Total Summary
    grand_total = total_dist + total_macro + total_bouzidi + total_geom + total_conn
    
    println("\n" * "="^70)
    println("                          MEMORY SUMMARY")
    println("="^70)
    @printf("  Distributions:      %7.1f MB  (%5.1f%%)\n", 
            total_dist, 100*total_dist/grand_total)
    @printf("  Macroscopic:        %7.1f MB  (%5.1f%%)\n", 
            total_macro, 100*total_macro/grand_total)
    @printf("  Geometry:           %7.1f MB  (%5.1f%%)\n", 
            total_geom, 100*total_geom/grand_total)
    @printf("  Connectivity:       %7.1f MB  (%5.1f%%)\n", 
            total_conn, 100*total_conn/grand_total)
    @printf("  Bouzidi IBM:        %7.1f MB  (%5.1f%%)\n", 
            total_bouzidi, 100*total_bouzidi/grand_total)
    println("-"^70)
    @printf("  TOTAL ALLOCATED:    %7.1f MB\n", grand_total)
    
    # Compare with CUDA driver report
    if CUDA.functional()
        free_mem, total_mem = CUDA.memory_info()
        used_mem = total_mem - free_mem
        used_mb = used_mem / 1024^2
        overhead = used_mb - grand_total
        
        println("-"^70)
        @printf("  CUDA REPORTED:      %7.1f MB (Includes driver + overhead)\n", used_mb)
        @printf("  Driver Overhead:    %7.1f MB\n", overhead)
        @printf("  Free VRAM:          %7.1f MB\n", free_mem/1024^2)
    end
    
    println("="^70)
    
    return (
        total = grand_total,
        distributions = total_dist,
        macroscopic = total_macro,
        geometry = total_geom,
        connectivity = total_conn,
        bouzidi = total_bouzidi
    )
end


"""
    estimate_mesh_capacity(target_vram_gb)

Estimate maximum mesh size for given VRAM budget.
"""
function estimate_mesh_capacity(target_vram_gb::Float64=8.0)
    # Approximate bytes per cell for D3Q27 AA-Pattern
    # f(27*4) + f_temp(27*4) + rho(4) + u(12) + u_temp(12) + obs(1) + wall(4) + sponge(4) ~ 253 bytes
    # Plus overhead ~ 300 bytes
    bytes_per_cell = 300.0 
    
    available_bytes = target_vram_gb * 1024^3
    capacity_cells = available_bytes / bytes_per_cell
    
    println("\n┌─ MESH CAPACITY ESTIMATE ($(target_vram_gb) GB VRAM) ────────────────────┐")
    @printf("│  Max capacity:       %6.1f M cells                                │\n", 
            capacity_cells/1e6)
    println("└───────────────────────────────────────────────────────────────────┘")
    
    return capacity_cells
end