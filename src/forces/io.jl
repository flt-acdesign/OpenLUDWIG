"""
IO.JL - Force/Coefficient Output Functions

Handles:
- VTK surface files for ParaView visualization
- CSV time-history files for coefficient tracking
- Summary printing
"""

using Printf
using WriteVTK

"""
    save_surface_vtk(filename, force_data, mesh)

Save surface pressure and shear stress data to a VTK file for ParaView.

The file includes:
- Pressure_Pa: Surface pressure [Pa]
- ShearX/Y/Z_Pa: Shear stress components [Pa]
- ShearMagnitude_Pa: |τ| [Pa]
- Normal: Surface normal vectors
- Area_m2: Triangle areas [m²]
- MappingQuality: 1.0 if mapped, 0.0 if not
"""
function save_surface_vtk(filename::String, force_data::ForceData, mesh::Geometry.SolverMesh)
    # Transfer from GPU
    p = Array(force_data.pressure_map)
    sx = Array(force_data.shear_x_map)
    sy = Array(force_data.shear_y_map)
    sz = Array(force_data.shear_z_map)
    
    n_tri = length(mesh.triangles)
    
    # Build point array (3 vertices per triangle)
    points = zeros(Float64, 3, n_tri * 3)
    cells = Vector{MeshCell}(undef, n_tri)
    
    idx = 1
    for i in 1:n_tri
        t = mesh.triangles[i]
        points[:, idx]   .= t[1]
        points[:, idx+1] .= t[2]
        points[:, idx+2] .= t[3]
        cells[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE, [idx, idx+1, idx+2])
        idx += 3
    end
    
    # Derived quantities
    shear_mag = sqrt.(sx.^2 .+ sy.^2 .+ sz.^2)
    mapping_quality = Float32[(abs(p[i]) > 1e-10 || abs(sx[i]) > 1e-10) ? 1.0f0 : 0.0f0 for i in 1:n_tri]
    
    # Normal vectors
    normals = zeros(Float32, 3, n_tri)
    for i in 1:n_tri
        normals[1, i] = Float32(mesh.normals[i][1])
        normals[2, i] = Float32(mesh.normals[i][2])
        normals[3, i] = Float32(mesh.normals[i][3])
    end
    
    # Areas
    areas = Float32[Float32(a) for a in mesh.areas]
    
    # Write VTK file
    vtk_file = vtk_grid(filename, points, cells)
    
    # Cell data (per triangle)
    vtk_cell_data(vtk_file, Float32.(p), "Pressure_Pa")
    vtk_cell_data(vtk_file, Float32.(sx), "ShearX_Pa")
    vtk_cell_data(vtk_file, Float32.(sy), "ShearY_Pa")
    vtk_cell_data(vtk_file, Float32.(sz), "ShearZ_Pa")
    vtk_cell_data(vtk_file, Float32.(shear_mag), "ShearMagnitude_Pa")
    vtk_cell_data(vtk_file, normals, "Normal")
    vtk_cell_data(vtk_file, areas, "Area_m2")
    vtk_cell_data(vtk_file, mapping_quality, "MappingQuality")
    
    vtk_save(vtk_file)
    
    # Summary
    n_mapped = count(mapping_quality .> 0.5f0)
    println("[VTK] Saved: $(filename).vtu ($n_mapped/$n_tri triangles mapped)")
end

"""
    write_force_csv_header(filepath)

Write header for force time-history CSV file.
"""
function write_force_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io, "Step,Time_s,U_inlet,Fx_N,Fy_N,Fz_N,Fx_p_N,Fx_v_N,Mx_Nm,My_Nm,Mz_Nm,Cd,Cl,Cs,Cmy")
    end
end

"""
    append_force_csv(filepath, step, time_phys, force_data, u_inlet)

Append one row to force time-history CSV file.
"""
function append_force_csv(filepath::String, step::Int, time_phys::Float64, 
                          force_data::ForceData, u_inlet::Float32)
    open(filepath, "a") do io
        @printf(io, "%d,%.6e,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f\n",
                step, time_phys, u_inlet,
                force_data.Fx, force_data.Fy, force_data.Fz,
                force_data.Fx_pressure, force_data.Fx_viscous,
                force_data.Mx, force_data.My, force_data.Mz,
                force_data.Cd, force_data.Cl, force_data.Cs, force_data.Cmy)
    end
end

"""
    print_force_summary(force_data)

Print formatted summary of aerodynamic forces and coefficients.
"""
function print_force_summary(force_data::ForceData)
    println("\n" * "="^60)
    println("         AERODYNAMIC FORCES SUMMARY")
    println("="^60)
    
    println("\nReference Values:")
    @printf("  ρ_ref  = %.4f kg/m³\n", force_data.rho_ref)
    @printf("  U_ref  = %.4f m/s\n", force_data.u_ref)
    @printf("  A_ref  = %.4f m²\n", force_data.area_ref)
    @printf("  L_ref  = %.4f m\n", force_data.chord_ref)
    
    q_inf = 0.5 * force_data.rho_ref * force_data.u_ref^2
    @printf("  q_∞    = %.4f Pa\n", q_inf)
    
    println("\nForces [N]:")
    @printf("  Fx (drag)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n", 
            force_data.Fx, force_data.Fx_pressure, force_data.Fx_viscous)
    @printf("  Fy (side)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            force_data.Fy, force_data.Fy_pressure, force_data.Fy_viscous)
    @printf("  Fz (lift)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            force_data.Fz, force_data.Fz_pressure, force_data.Fz_viscous)
    
    println("\nMoments [N·m]:")
    @printf("  Mx (roll)  = %+.4e\n", force_data.Mx)
    @printf("  My (pitch) = %+.4e\n", force_data.My)
    @printf("  Mz (yaw)   = %+.4e\n", force_data.Mz)
    
    println("\nCoefficients:")
    @printf("  Cd = %+.6f\n", force_data.Cd)
    @printf("  Cl = %+.6f\n", force_data.Cl)
    @printf("  Cs = %+.6f\n", force_data.Cs)
    @printf("  Cmy = %+.6f\n", force_data.Cmy)
    
    # Pressure vs viscous drag breakdown
    if abs(force_data.Fx) > 1e-10
        p_frac = abs(force_data.Fx_pressure) / abs(force_data.Fx) * 100
        v_frac = abs(force_data.Fx_viscous) / abs(force_data.Fx) * 100
        @printf("\nDrag breakdown: %.1f%% pressure, %.1f%% viscous\n", p_frac, v_frac)
    end
    
    println("="^60 * "\n")
end

"""
    export_surface_loads_csv(filename, force_data, mesh, mesh_offset)

Export per-triangle surface loads in CSV format for external FEA tools.

Columns: triangle_id, center_xyz, normal_xyz, area, pressure, shear_xyz
"""
function export_surface_loads_csv(filename::String, force_data::ForceData, 
                                   mesh::Geometry.SolverMesh, mesh_offset::Tuple{Float64,Float64,Float64})
    p = Array(force_data.pressure_map)
    sx = Array(force_data.shear_x_map)
    sy = Array(force_data.shear_y_map)
    sz = Array(force_data.shear_z_map)
    
    open(filename, "w") do io
        println(io, "triangle_id,cx,cy,cz,nx,ny,nz,area_m2,pressure_Pa,shear_x_Pa,shear_y_Pa,shear_z_Pa")
        
        for i in 1:length(mesh.triangles)
            c = mesh.centers[i]
            n = mesh.normals[i]
            a = mesh.areas[i]
            
            @printf(io, "%d,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    i, c[1]+mesh_offset[1], c[2]+mesh_offset[2], c[3]+mesh_offset[3],
                    n[1], n[2], n[3], a,
                    p[i], sx[i], sy[i], sz[i])
        end
    end
    
    println("[CSV] Exported surface loads: $filename")
end