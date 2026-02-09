# FILE: src/forces/io.jl
using Printf
using WriteVTK

function write_force_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io, "Step,Time_phys_s,U_inlet_lat,Fx_N,Fy_N,Fz_N,Fx_p,Fy_p,Fz_p,Fx_v,Fy_v,Fz_v,Mx,My,Mz,Cd,Cl,Cs,Cmx,Cmy,Cmz")
    end
end

function append_force_csv(filepath::String, step::Int, time_phys::Float64,
                          fd::ForceData, u_inlet::Float32)
    open(filepath, "a") do io
        @printf(io, "%d,%.6e,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                step, time_phys, u_inlet,
                fd.Fx, fd.Fy, fd.Fz,
                fd.Fx_pressure, fd.Fy_pressure, fd.Fz_pressure,
                fd.Fx_viscous, fd.Fy_viscous, fd.Fz_viscous,
                fd.Mx, fd.My, fd.Mz,
                fd.Cd, fd.Cl, fd.Cs,
                fd.Cmx, fd.Cmy, fd.Cmz)
    end
end

function print_force_summary(fd::ForceData; re_number::Float64=0.0,
                              ma_physical::Float64=0.0, ma_lattice::Float64=0.0,
                              compressibility_correction::Bool=false)
    q_inf = 0.5 * fd.rho_ref * fd.u_ref^2
    F_ref = q_inf * fd.area_ref

    Cdp = F_ref > 1e-10 ? fd.Fx_pressure / F_ref : 0.0
    Cdv = F_ref > 1e-10 ? fd.Fx_viscous  / F_ref : 0.0

    println()
    println("============================================================")
    println("          FINAL AERODYNAMIC FORCES")
    if compressibility_correction
        println("          (Compressible solver: O(u⁴) equilibrium + DDF)")
    end
    println("============================================================")
    println()
    println("Reference Values:")
    @printf("  Re     = %.2e\n", re_number)
    @printf("  Ma     = %.4f (physical), %.4f (lattice)\n", ma_physical, ma_lattice)
    @printf("  ρ_ref  = %.4f kg/m³\n", fd.rho_ref)
    @printf("  U_ref  = %.4f m/s\n", fd.u_ref)
    @printf("  A_ref  = %.4f m²\n", fd.area_ref)
    @printf("  L_ref  = %.4f m\n", fd.chord_ref)
    @printf("  q_∞    = %.4f Pa\n", q_inf)
    println()
    println("Forces [N]:")
    @printf("  Fx (drag)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            fd.Fx, fd.Fx_pressure, fd.Fx_viscous)
    @printf("  Fy (side)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            fd.Fy, fd.Fy_pressure, fd.Fy_viscous)
    @printf("  Fz (lift)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n",
            fd.Fz, fd.Fz_pressure, fd.Fz_viscous)
    println()
    println("Moments [N·m]:")
    @printf("  Mx (roll)  = %+.4e\n", fd.Mx)
    @printf("  My (pitch) = %+.4e\n", fd.My)
    @printf("  Mz (yaw)   = %+.4e\n", fd.Mz)
    println()
    println("Coefficients:")
    @printf("  Cd  = %+.6f  (Cdp = %+.6f, Cdv = %+.6f)\n", fd.Cd, Cdp, Cdv)
    @printf("  Cl  = %+.6f\n", fd.Cl)
    @printf("  Cs  = %+.6f\n", fd.Cs)
    @printf("  Cmx = %+.6f\n", fd.Cmx)
    @printf("  Cmy = %+.6f\n", fd.Cmy)
    @printf("  Cmz = %+.6f\n", fd.Cmz)
    println("============================================================")
end

# Modified to accept params and apply offset
function save_surface_vtk(filename::String, force_data::ForceData,
                          mesh::Geometry.SolverMesh, params;
                          object_names=nothing)

    n_tri = length(mesh.triangles)
    ox = params.mesh_offset[1]
    oy = params.mesh_offset[2]
    oz = params.mesh_offset[3]

    n_pts = 3 * n_tri
    points = Matrix{Float64}(undef, 3, n_pts)
    cells  = Vector{MeshCell}(undef, n_tri)

    for i in 1:n_tri
        tri = mesh.triangles[i]
        for (vi, v) in enumerate(tri)
            pidx = 3 * (i - 1) + vi
            # Apply domain offset here
            points[1, pidx] = v[1] + ox
            points[2, pidx] = v[2] + oy
            points[3, pidx] = v[3] + oz
        end
        base = 3 * (i - 1)
        cells[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE,
                             (base + 1, base + 2, base + 3))
    end

    p_cpu  = Array(force_data.pressure_map)
    sx_cpu = Array(force_data.shear_x_map)
    sy_cpu = Array(force_data.shear_y_map)
    sz_cpu = Array(force_data.shear_z_map)

    shear_mag = sqrt.(sx_cpu .^ 2 .+ sy_cpu .^ 2 .+ sz_cpu .^ 2)

    normals = Matrix{Float64}(undef, 3, n_tri)
    for i in 1:n_tri
        normals[1, i] = mesh.normals[i][1]
        normals[2, i] = mesh.normals[i][2]
        normals[3, i] = mesh.normals[i][3]
    end

    shear_vec = Matrix{Float32}(undef, 3, n_tri)
    shear_vec[1, :] .= sx_cpu
    shear_vec[2, :] .= sy_cpu
    shear_vec[3, :] .= sz_cpu

    try
        vtk_grid(filename, points, cells; compress=true, append=false) do vtk
            vtk["Pressure"]      = p_cpu
            vtk["ShearStress"]   = shear_vec
            vtk["ShearMag"]      = shear_mag
            vtk["Normal"]        = normals
            vtk["Area"]          = Float32.(mesh.areas)
            if object_names !== nothing
                # Map unique object names to integer IDs for ParaView filtering
                unique_names = unique(object_names)
                name_to_id = Dict(n => Int32(i) for (i, n) in enumerate(unique_names))
                obj_ids = Int32[name_to_id[object_names[j]] for j in 1:n_tri]
                vtk["ObjectID"] = obj_ids
            end
        end
    catch e
        println("[Error] Surface VTK write failed: $e")
    end
end

function export_net_triangle_forces_csv(filename::String,
                                        net_Fx::Vector{Float32},
                                        net_Fy::Vector{Float32},
                                        net_Fz::Vector{Float32},
                                        gpu_mesh, mesh_offset;
                                        unrotated_centers=nothing,
                                        alpha_deg::Float64=0.0,
                                        beta_deg::Float64=0.0,
                                        object_names=nothing)

    n_tri = length(net_Fx)

    # Use unrotated (subdivided-but-not-rotated) centroids when available.
    # These are in original STL coordinates — no domain offset applied.
    # When unrotated coords are used, force vectors must also be inverse-rotated
    # back to the original STL frame so positions and forces are consistent.
    # Otherwise fall back to the rotated GPU mesh centroids + domain offset.
    use_unrotated = unrotated_centers !== nothing

    if use_unrotated
        cx_cpu = Float32[c[1] for c in unrotated_centers]
        cy_cpu = Float32[c[2] for c in unrotated_centers]
        cz_cpu = Float32[c[3] for c in unrotated_centers]
    else
        cx_cpu = Array(gpu_mesh.centers_x)
        cy_cpu = Array(gpu_mesh.centers_y)
        cz_cpu = Array(gpu_mesh.centers_z)
    end

    ox = use_unrotated ? 0.0f0 : Float32(mesh_offset[1])
    oy = use_unrotated ? 0.0f0 : Float32(mesh_offset[2])
    oz = use_unrotated ? 0.0f0 : Float32(mesh_offset[3])

    # Build inverse rotation matrix R^T to transform forces back to the
    # original STL coordinate system (only when using unrotated centroids).
    # The forward rotation in geometry.jl is  R = Rz * Ry  (column-major SMatrix),
    # which gives the matrix:
    #   R = [ ca*cb    sb   -sa*cb ]
    #       [-ca*sb    cb    sa*sb ]
    #       [   sa      0      ca  ]
    # The inverse is R^T:
    #   R^T = [ ca*cb  -ca*sb   sa ]
    #         [   sb      cb     0 ]
    #         [-sa*cb   sa*sb   ca ]
    need_inv_rotate = use_unrotated && (alpha_deg != 0.0 || beta_deg != 0.0)

    if need_inv_rotate
        alpha = deg2rad(alpha_deg)
        beta  = deg2rad(beta_deg)
        ca, sa = cos(alpha), sin(alpha)
        cb, sb = cos(beta),  sin(beta)

        # R^T entries (row-major: r_ij = R^T[i,j])
        r11 = Float32( ca*cb);  r12 = Float32(-ca*sb);  r13 = Float32( sa)
        r21 = Float32( sb);     r22 = Float32( cb);     r23 = Float32(0.0)
        r31 = Float32(-sa*cb);  r32 = Float32( sa*sb);  r33 = Float32( ca)
    end

    out_dir = dirname(filename)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    open(filename, "w") do f
        println(f, "triangle_id,object_name,cx,cy,cz,Fx,Fy,Fz")

        for i in 1:n_tri
            fx = net_Fx[i]
            fy = net_Fy[i]
            fz = net_Fz[i]

            if need_inv_rotate
                # Apply R^{-1} to force vector
                fx_rot = r11*fx + r12*fy + r13*fz
                fy_rot = r21*fx + r22*fy + r23*fz
                fz_rot = r31*fx + r32*fy + r33*fz
                fx = fx_rot
                fy = fy_rot
                fz = fz_rot
            end

            obj_name = object_names !== nothing ? object_names[i] : "default"

            @printf(f, "%d,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    i, obj_name,
                    cx_cpu[i] + ox,
                    cy_cpu[i] + oy,
                    cz_cpu[i] + oz,
                    fx, fy, fz)
        end
    end

    println("[NetForce] Exported $(n_tri) triangle forces to $(filename)")
end

"""
    export_surface_pressure_bin(filename, force_data, mesh, params; object_names=nothing)

Export surface pressure data as a compact binary file for web visualization.
The file contains the full triangle mesh with per-face pressure and shear magnitude,
readable by postv8.html (BabylonJS-based pressure viewer).

Binary format (little-endian):
  [Header]
    4 bytes  : magic "LBMP"
    4 bytes  : version (UInt32 = 1)
    4 bytes  : n_triangles (UInt32)
    4 bytes  : n_objects (UInt32)
  [Object name table]  — for each object:
    4 bytes  : name length L (UInt32)
    L bytes  : UTF-8 name string
  [Per-triangle data]  — all arrays are contiguous Float32 / Int32:
    n_tri × 9 Float32 : vertices (v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z)
    n_tri × 3 Float32 : normals  (nx, ny, nz)
    n_tri     Float32 : pressure (gauge, physical Pa)
    n_tri     Float32 : shear magnitude (physical Pa)
    n_tri     Float32 : Cp (pressure coefficient)
    n_tri     Int32   : object ID (1-based index into name table)
"""
function export_surface_pressure_bin(filename::String, force_data::ForceData,
                                      mesh::Geometry.SolverMesh, params;
                                      object_names=nothing,
                                      use_domain_offset::Bool=true)

    n_tri = length(mesh.triangles)
    ox = use_domain_offset ? Float32(params.mesh_offset[1]) : 0.0f0
    oy = use_domain_offset ? Float32(params.mesh_offset[2]) : 0.0f0
    oz = use_domain_offset ? Float32(params.mesh_offset[3]) : 0.0f0

    # Build object name table
    if object_names !== nothing
        unique_names = unique(object_names)
    else
        unique_names = ["default"]
    end
    n_objects = length(unique_names)
    name_to_id = Dict(n => Int32(i) for (i, n) in enumerate(unique_names))

    # Copy GPU data to CPU
    p_cpu  = Array(force_data.pressure_map)
    sx_cpu = Array(force_data.shear_x_map)
    sy_cpu = Array(force_data.shear_y_map)
    sz_cpu = Array(force_data.shear_z_map)
    shear_mag = sqrt.(sx_cpu .^ 2 .+ sy_cpu .^ 2 .+ sz_cpu .^ 2)

    # Compute Cp = p_gauge / q_inf
    q_inf = Float32(0.5 * force_data.rho_ref * force_data.u_ref^2)
    Cp = q_inf > Float32(1e-10) ? p_cpu ./ q_inf : zeros(Float32, n_tri)

    out_dir = dirname(filename)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    open(filename, "w") do io
        # Header
        write(io, UInt8.([0x4C, 0x42, 0x4D, 0x50]))  # "LBMP"
        write(io, UInt32(1))                            # version
        write(io, UInt32(n_tri))
        write(io, UInt32(n_objects))

        # Object name table
        for name in unique_names
            name_bytes = Vector{UInt8}(codeunits(name))
            write(io, UInt32(length(name_bytes)))
            write(io, name_bytes)
        end

        # Vertices: n_tri × 9 Float32 (with domain offset applied)
        for i in 1:n_tri
            tri = mesh.triangles[i]
            for v in tri
                write(io, Float32(v[1]) + ox)
                write(io, Float32(v[2]) + oy)
                write(io, Float32(v[3]) + oz)
            end
        end

        # Normals: n_tri × 3 Float32
        for i in 1:n_tri
            write(io, Float32(mesh.normals[i][1]))
            write(io, Float32(mesh.normals[i][2]))
            write(io, Float32(mesh.normals[i][3]))
        end

        # Pressure: n_tri Float32
        for i in 1:n_tri
            write(io, p_cpu[i])
        end

        # Shear magnitude: n_tri Float32
        for i in 1:n_tri
            write(io, shear_mag[i])
        end

        # Cp: n_tri Float32
        for i in 1:n_tri
            write(io, Cp[i])
        end

        # Object IDs: n_tri Int32
        for i in 1:n_tri
            obj_name = object_names !== nothing ? object_names[i] : "default"
            write(io, name_to_id[obj_name])
        end
    end

    file_size_mb = filesize(filename) / (1024^2)
    @printf("[Surface] Exported binary pressure map: %s (%.1f MB, %d triangles)\n",
            filename, file_size_mb, n_tri)
end

function export_net_triangle_forces_vtk(filename::String,
                                        net_Fx::Vector{Float32},
                                        net_Fy::Vector{Float32},
                                        net_Fz::Vector{Float32},
                                        gpu_mesh, mesh_offset)

    n_tri = length(net_Fx)

    cx_cpu = Array(gpu_mesh.centers_x)
    cy_cpu = Array(gpu_mesh.centers_y)
    cz_cpu = Array(gpu_mesh.centers_z)
    nx_cpu = Array(gpu_mesh.normals_x)
    ny_cpu = Array(gpu_mesh.normals_y)
    nz_cpu = Array(gpu_mesh.normals_z)
    areas_cpu = Array(gpu_mesh.areas)

    ox = Float64(mesh_offset[1])
    oy = Float64(mesh_offset[2])
    oz = Float64(mesh_offset[3])

    vtp_filename = filename * ".vtp"

    out_dir = dirname(vtp_filename)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    open(vtp_filename, "w") do f
        println(f, "<?xml version=\"1.0\"?>")
        println(f, "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">")
        println(f, "  <PolyData>")
        println(f, "    <Piece NumberOfPoints=\"$(n_tri)\" NumberOfVerts=\"$(n_tri)\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">")

        println(f, "      <Points>")
        println(f, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.8e %.8e %.8e\n",
                    Float64(cx_cpu[i]) + ox,
                    Float64(cy_cpu[i]) + oy,
                    Float64(cz_cpu[i]) + oz)
        end
        println(f, "        </DataArray>")
        println(f, "      </Points>")

        println(f, "      <Verts>")
        println(f, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")
        for i in 0:(n_tri - 1)
            @printf(f, "          %d\n", i)
        end
        println(f, "        </DataArray>")
        println(f, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %d\n", i)
        end
        println(f, "        </DataArray>")
        println(f, "      </Verts>")

        println(f, "      <PointData Vectors=\"NetForce\">")

        println(f, "        <DataArray type=\"Float32\" Name=\"NetForce\" NumberOfComponents=\"3\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.6e %.6e %.6e\n", net_Fx[i], net_Fy[i], net_Fz[i])
        end
        println(f, "        </DataArray>")

        println(f, "        <DataArray type=\"Float32\" Name=\"NetForceMagnitude\" format=\"ascii\">")
        for i in 1:n_tri
            mag = sqrt(net_Fx[i]^2 + net_Fy[i]^2 + net_Fz[i]^2)
            @printf(f, "          %.6e\n", mag)
        end
        println(f, "        </DataArray>")

        println(f, "        <DataArray type=\"Float32\" Name=\"Normal\" NumberOfComponents=\"3\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.6e %.6e %.6e\n", nx_cpu[i], ny_cpu[i], nz_cpu[i])
        end
        println(f, "        </DataArray>")

        println(f, "        <DataArray type=\"Float32\" Name=\"Area\" format=\"ascii\">")
        for i in 1:n_tri
            @printf(f, "          %.6e\n", areas_cpu[i])
        end
        println(f, "        </DataArray>")

        println(f, "      </PointData>")

        println(f, "    </Piece>")
        println(f, "  </PolyData>")
        println(f, "</VTKFile>")
    end

    println("[NetForce] Exported VTK PolyData to $(vtp_filename)")
end