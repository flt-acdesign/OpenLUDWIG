#
module Geometry

using LinearAlgebra
using Printf
using StaticArrays
using Base.Threads
using WriteVTK
using CUDA
using Adapt
using KernelAbstractions

export load_mesh, compute_mesh_bounds, GPUMesh, upload_mesh_to_gpu
export SolverMesh, subdivide_mesh, write_stl_binary, rotate_mesh

struct SolverMesh
    triangles::Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}
    min_bounds::Tuple{Float64, Float64, Float64}
    max_bounds::Tuple{Float64, Float64, Float64}
    
    normals::Vector{SVector{3, Float64}}
    areas::Vector{Float64}
    centers::Vector{SVector{3, Float64}}
end

struct GPUMesh{A}
    n_triangles::Int
    centers_x::A
    centers_y::A
    centers_z::A
    normals_x::A
    normals_y::A
    normals_z::A
    areas::A
end

function Adapt.adapt_structure(to, mesh::GPUMesh)
    GPUMesh(
        mesh.n_triangles,
        adapt(to, mesh.centers_x),
        adapt(to, mesh.centers_y),
        adapt(to, mesh.centers_z),
        adapt(to, mesh.normals_x),
        adapt(to, mesh.normals_y),
        adapt(to, mesh.normals_z),
        adapt(to, mesh.areas)
    )
end

function upload_mesh_to_gpu(mesh::SolverMesh, backend)
    n = length(mesh.triangles)
    
    cx = Float32[c[1] for c in mesh.centers]
    cy = Float32[c[2] for c in mesh.centers]
    cz = Float32[c[3] for c in mesh.centers]
    
    nx = Float32[n[1] for n in mesh.normals]
    ny = Float32[n[2] for n in mesh.normals]
    nz = Float32[n[3] for n in mesh.normals]
    
    areas = Float32[a for a in mesh.areas]
    
    return GPUMesh(
        n,
        adapt(backend, cx),
        adapt(backend, cy),
        adapt(backend, cz),
        adapt(backend, nx),
        adapt(backend, ny),
        adapt(backend, nz),
        adapt(backend, areas)
    )
end

function compute_geometry_properties(triangles)
    n = length(triangles)
    normals = Vector{SVector{3, Float64}}(undef, n)
    areas = Vector{Float64}(undef, n)
    centers = Vector{SVector{3, Float64}}(undef, n)

    @threads for i in 1:n
        v1 = SVector(triangles[i][1])
        v2 = SVector(triangles[i][2])
        v3 = SVector(triangles[i][3])

        edge1 = v2 - v1
        edge2 = v3 - v1
        cross_prod = cross(edge1, edge2)
        
        area = 0.5 * norm(cross_prod)
        if area > 1e-12
            normal = cross_prod / (2.0 * area)
        else
            normal = SVector(0.0, 0.0, 0.0)
        end
        
        normals[i] = normal
        areas[i] = area
        centers[i] = (v1 + v2 + v3) / 3.0
    end

    return normals, areas, centers
end

# ──────────────────────────────────────────────────────────────────────────────
#  Triangle subdivision (longest-edge bisection, iterative)
# ──────────────────────────────────────────────────────────────────────────────

"""
    subdivide_triangle(tri, min_size) -> Vector of triangles

Iteratively subdivide a triangle by bisecting the longest edge until every
edge is ≤ `min_size`.  Each bisection splits the longest edge at its midpoint,
connecting the midpoint to the opposite vertex to produce 2 child triangles.
This guarantees the longest edge halves each level, so it converges in
O(log₂(L/min_size)) levels with O(2^levels) output triangles.

Uses an explicit worklist to avoid stack overflow.
"""
function subdivide_triangle(tri::Tuple{Tuple{Float64,Float64,Float64},
                                       Tuple{Float64,Float64,Float64},
                                       Tuple{Float64,Float64,Float64}},
                            min_size::Float64)
    T = typeof(tri)
    result = T[]
    worklist = T[tri]

    while !isempty(worklist)
        t = pop!(worklist)

        v1 = SVector(t[1])
        v2 = SVector(t[2])
        v3 = SVector(t[3])

        e1 = norm(v2 - v1)   # edge v1-v2
        e2 = norm(v3 - v2)   # edge v2-v3
        e3 = norm(v1 - v3)   # edge v3-v1

        longest = max(e1, e2, e3)

        if longest <= min_size
            push!(result, t)
        else
            # Bisect the longest edge at its midpoint
            if e1 >= e2 && e1 >= e3
                # Longest edge is v1-v2 → midpoint m, opposite vertex v3
                m = Tuple((v1 + v2) / 2.0)
                push!(worklist, (t[1], m,    t[3]))   # (v1, m, v3)
                push!(worklist, (m,    t[2], t[3]))   # (m, v2, v3)
            elseif e2 >= e1 && e2 >= e3
                # Longest edge is v2-v3 → midpoint m, opposite vertex v1
                m = Tuple((v2 + v3) / 2.0)
                push!(worklist, (t[2], m,    t[1]))   # (v2, m, v1)
                push!(worklist, (m,    t[3], t[1]))   # (m, v3, v1)
            else
                # Longest edge is v3-v1 → midpoint m, opposite vertex v2
                m = Tuple((v3 + v1) / 2.0)
                push!(worklist, (t[3], m,    t[2]))   # (v3, m, v2)
                push!(worklist, (m,    t[1], t[2]))   # (m, v1, v2)
            end
        end
    end

    return result
end

"""
    subdivide_mesh(mesh, min_size) -> SolverMesh

Subdivide every triangle in `mesh` whose longest edge exceeds `min_size`
using longest-edge bisection.
Returns a new SolverMesh with inherited parent normals and recomputed areas/centers.
"""
function subdivide_mesh(mesh::SolverMesh, min_size::Float64)
    if min_size <= 0.0
        return mesh
    end

    n_orig = length(mesh.triangles)

    # Estimate upper bound on output triangles
    max_edge = 0.0
    for tri in mesh.triangles
        v1 = SVector(tri[1]); v2 = SVector(tri[2]); v3 = SVector(tri[3])
        max_edge = max(max_edge, norm(v2-v1), norm(v3-v2), norm(v1-v3))
    end
    max_levels = max_edge > min_size ? ceil(Int, log2(max_edge / min_size)) : 0
    est_factor = 2^max_levels
    est_total  = n_orig * est_factor

    @printf("[Geometry] Subdividing %d triangles (min facet size: %.4f m)\n", n_orig, min_size)
    @printf("[Geometry]   Largest edge: %.4f m → up to %d bisection levels\n", max_edge, max_levels)
    @printf("[Geometry]   Estimated output: ~%d triangles (upper bound)\n", est_total)

    MAX_TRIANGLES = 50_000_000   # 50M safety cap
    if est_total > MAX_TRIANGLES
        @printf("[Geometry] ⚠ Estimated triangle count (%d) exceeds safety cap (%d).\n", est_total, MAX_TRIANGLES)
        @printf("[Geometry]   Consider increasing minimum_facet_size. Proceeding anyway...\n")
    end

    new_triangles = Vector{eltype(mesh.triangles)}()
    new_normals   = Vector{SVector{3, Float64}}()
    sizehint!(new_triangles, min(est_total, MAX_TRIANGLES))
    sizehint!(new_normals,   min(est_total, MAX_TRIANGLES))

    for (i, tri) in enumerate(mesh.triangles)
        parent_normal = mesh.normals[i]
        children = subdivide_triangle(tri, min_size)
        append!(new_triangles, children)

        # All children inherit the parent's normal — no recomputation from cross product
        for _ in 1:length(children)
            push!(new_normals, parent_normal)
        end

        if length(new_triangles) > MAX_TRIANGLES
            error("[Geometry] Subdivision exceeded $(MAX_TRIANGLES) triangles. " *
                  "Increase minimum_facet_size or reduce mesh complexity.")
        end
    end

    # Recompute areas and centers (these are geometry-dependent, not orientation-dependent)
    n_new = length(new_triangles)
    new_areas   = Vector{Float64}(undef, n_new)
    new_centers = Vector{SVector{3, Float64}}(undef, n_new)
    @threads for i in 1:n_new
        v1 = SVector(new_triangles[i][1])
        v2 = SVector(new_triangles[i][2])
        v3 = SVector(new_triangles[i][3])
        new_areas[i]   = 0.5 * norm(cross(v2 - v1, v3 - v1))
        new_centers[i] = (v1 + v2 + v3) / 3.0
    end

    # Recompute bounds from the new triangle set
    min_x, min_y, min_z =  Inf,  Inf,  Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    for tri in new_triangles
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    @printf("[Geometry] Subdivision complete: %d → %d triangles (%.1fx)\n",
            n_orig, length(new_triangles), length(new_triangles)/n_orig)

    return SolverMesh(new_triangles,
                      (min_x, min_y, min_z),
                      (max_x, max_y, max_z),
                      new_normals, new_areas, new_centers)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Mesh rotation (alpha around Y, then beta around Z)
# ──────────────────────────────────────────────────────────────────────────────

"""
    rotate_mesh(mesh, alpha_deg, beta_deg) -> SolverMesh

Rotate the mesh: first by `alpha_deg` around the Y axis, then by `beta_deg`
around the Z axis.  Both angles are in degrees.  The rotation is applied about
the mesh centroid so that the model stays centred.

Returns a new SolverMesh with rotated triangles, normals, centers, and updated bounds.
"""
function rotate_mesh(mesh::SolverMesh, alpha_deg::Float64, beta_deg::Float64)
    if alpha_deg == 0.0 && beta_deg == 0.0
        return mesh
    end

    # Build combined rotation matrix  R = Rz(beta) * Ry(alpha)
    alpha = deg2rad(alpha_deg)
    beta  = deg2rad(beta_deg)

    ca, sa = cos(alpha), sin(alpha)
    cb, sb = cos(beta),  sin(beta)

    # Ry(alpha) — rotation around Y
    Ry = SMatrix{3,3,Float64}(
         ca,  0.0, sa,
         0.0, 1.0, 0.0,
        -sa,  0.0, ca
    )

    # Rz(beta)  — rotation around Z
    Rz = SMatrix{3,3,Float64}(
        cb, -sb, 0.0,
        sb,  cb, 0.0,
        0.0, 0.0, 1.0
    )

    R = Rz * Ry   # apply Ry first, then Rz

    # Rotation centre = mesh centroid
    cx = (mesh.min_bounds[1] + mesh.max_bounds[1]) / 2.0
    cy = (mesh.min_bounds[2] + mesh.max_bounds[2]) / 2.0
    cz = (mesh.min_bounds[3] + mesh.max_bounds[3]) / 2.0
    origin = SVector(cx, cy, cz)

    # Helper: rotate a point about origin
    @inline rot_pt(p) = Tuple(R * (SVector(p) - origin) + origin)

    n_tri = length(mesh.triangles)
    new_tris    = Vector{eltype(mesh.triangles)}(undef, n_tri)
    new_normals = Vector{SVector{3, Float64}}(undef, n_tri)
    new_centers = Vector{SVector{3, Float64}}(undef, n_tri)
    new_areas   = Vector{Float64}(undef, n_tri)

    @threads for i in 1:n_tri
        v1r = rot_pt(mesh.triangles[i][1])
        v2r = rot_pt(mesh.triangles[i][2])
        v3r = rot_pt(mesh.triangles[i][3])
        new_tris[i] = (v1r, v2r, v3r)

        # Rotate normal (pure direction — no translation)
        new_normals[i] = R * mesh.normals[i]

        # Recompute area & centre from rotated vertices
        sv1 = SVector(v1r); sv2 = SVector(v2r); sv3 = SVector(v3r)
        new_areas[i]   = 0.5 * norm(cross(sv2 - sv1, sv3 - sv1))
        new_centers[i] = (sv1 + sv2 + sv3) / 3.0
    end

    # Recompute bounds
    min_x, min_y, min_z =  Inf,  Inf,  Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    for tri in new_tris
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    @printf("[Geometry] Rotated mesh: α=%.2f° (Y), β=%.2f° (Z)\n", alpha_deg, beta_deg)
    @printf("[Geometry] New bounds: [%.3f, %.3f] x [%.3f, %.3f] x [%.3f, %.3f]\n",
            min_x, max_x, min_y, max_y, min_z, max_z)

    return SolverMesh(new_tris,
                      (min_x, min_y, min_z),
                      (max_x, max_y, max_z),
                      new_normals, new_areas, new_centers)
end

# ──────────────────────────────────────────────────────────────────────────────
#  Binary STL writer
# ──────────────────────────────────────────────────────────────────────────────

"""
    write_stl_binary(filename, mesh)

Write the mesh to a binary STL file.
"""
function write_stl_binary(filename::String, mesh::SolverMesh)
    open(filename, "w") do io
        # 80-byte header
        header = zeros(UInt8, 80)
        msg = Vector{UInt8}(codeunits("LBM Solver - Subdivided STL"))
        header[1:length(msg)] .= msg
        write(io, header)

        # Triangle count
        write(io, UInt32(length(mesh.triangles)))

        for (i, tri) in enumerate(mesh.triangles)
            n = mesh.normals[i]
            # Normal
            write(io, Float32(n[1])); write(io, Float32(n[2])); write(io, Float32(n[3]))
            # Vertices
            for v in tri
                write(io, Float32(v[1])); write(io, Float32(v[2])); write(io, Float32(v[3]))
            end
            # Attribute byte count
            write(io, UInt16(0))
        end
    end
    println("[Geometry] Wrote subdivided STL: $filename ($(length(mesh.triangles)) triangles)")
end

# ──────────────────────────────────────────────────────────────────────────────
#  STL loaders (winding-corrected using STL facet normals)
# ──────────────────────────────────────────────────────────────────────────────

function parse_binary_stl(io::IO, scale::Float64)
    skip(io, 80)
    count = read(io, UInt32)
    triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}(undef, count)
    n_flipped = 0
    for i in 1:count
        # ── Read the facet normal stored in the STL file ──
        fnx = Float64(read(io, Float32))
        fny = Float64(read(io, Float32))
        fnz = Float64(read(io, Float32))

        x1 = Float64(read(io, Float32)) * scale
        y1 = Float64(read(io, Float32)) * scale
        z1 = Float64(read(io, Float32)) * scale
        x2 = Float64(read(io, Float32)) * scale
        y2 = Float64(read(io, Float32)) * scale
        z2 = Float64(read(io, Float32)) * scale
        x3 = Float64(read(io, Float32)) * scale
        y3 = Float64(read(io, Float32)) * scale
        z3 = Float64(read(io, Float32)) * scale
        skip(io, 2)

        # ── Check if vertex winding agrees with the stored normal ──
        # cross(v2-v1, v3-v1)  should point in the same direction as (fnx,fny,fnz)
        e1x = x2-x1; e1y = y2-y1; e1z = z2-z1
        e2x = x3-x1; e2y = y3-y1; e2z = z3-z1
        cx = e1y*e2z - e1z*e2y
        cy = e1z*e2x - e1x*e2z
        cz = e1x*e2y - e1y*e2x

        # dot(cross, stl_normal) < 0  →  winding is flipped  →  swap v2 ↔ v3
        if (cx*fnx + cy*fny + cz*fnz) < 0.0
            triangles[i] = ((x1,y1,z1), (x3,y3,z3), (x2,y2,z2))
            n_flipped += 1
        else
            triangles[i] = ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3))
        end
    end
    if n_flipped > 0
        @printf("[Geometry] Fixed vertex winding on %d / %d triangles (normals now consistent)\n", n_flipped, count)
    end
    return triangles
end

function parse_ascii_stl(filename::String, scale::Float64)
    triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}()
    current_tri = Vector{Tuple{Float64,Float64,Float64}}()
    for line in eachline(filename)
        s = strip(line)
        if startswith(s, "vertex")
            parts = split(s)
            if length(parts) >= 4
                x = parse(Float64, parts[2]) * scale
                y = parse(Float64, parts[3]) * scale
                z = parse(Float64, parts[4]) * scale
                push!(current_tri, (x, y, z))
            end
        elseif startswith(s, "endloop")
            if length(current_tri) == 3
                push!(triangles, (current_tri[1], current_tri[2], current_tri[3]))
            end
            empty!(current_tri)
        end
    end
    return triangles
end

"""
    load_mesh(filename; scale=1.0, min_facet_size=0.0)

Load an STL file. If `min_facet_size > 0`, subdivide triangles so that no edge
exceeds that length, then write the subdivided mesh as `subdivided_<original>`.
"""
function load_mesh(filename::String; scale=1.0, min_facet_size=0.0)
    if !isfile(filename); error("STL file not found: $filename"); end
    println("[Geometry] Loading STL from: $filename")
    
    triangles = nothing
    is_binary = true
    
    open(filename, "r") do io
        if filesize(filename) < 84
            is_binary = false
        else
            header = String(read(io, 5))
            if startswith(lowercase(header), "solid")
                seek(io, 80)
                count = read(io, UInt32)
                if filesize(filename) != 84 + count * 50
                    is_binary = false
                end
            end
        end
    end
    
    if is_binary
        open(filename, "r") do io; triangles = parse_binary_stl(io, scale); end
        println("[Geometry] Format: Binary STL")
    else
        triangles = parse_ascii_stl(filename, scale)
        println("[Geometry] Format: ASCII STL")
    end

    if isempty(triangles); error("No triangles loaded."); end
    
    min_x, min_y, min_z = Inf, Inf, Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf

    for tri in triangles
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end
    
    normals, areas, centers = compute_geometry_properties(triangles)

    println("[Geometry] Loaded $(length(triangles)) triangles.")
    println("[Geometry] Bounds: [$(round(min_x,digits=3)), $(round(max_x,digits=3))] x [$(round(min_y,digits=3)), $(round(max_y,digits=3))] x [$(round(min_z,digits=3)), $(round(max_z,digits=3))]")

    mesh = SolverMesh(triangles, (min_x, min_y, min_z), (max_x, max_y, max_z), normals, areas, centers)

    # ── Subdivision ──────────────────────────────────────────────────────────
    if min_facet_size > 0.0
        mesh = subdivide_mesh(mesh, Float64(min_facet_size))

        # Write the subdivided STL next to the original
        dir  = dirname(filename)
        base = basename(filename)
        out_file = joinpath(dir, "subdivided_" * base)
        write_stl_binary(out_file, mesh)
    end

    return mesh
end

function compute_mesh_bounds(mesh::SolverMesh)
    return (min_bounds=mesh.min_bounds, max_bounds=mesh.max_bounds)
end

"""
    merge_meshes(meshes::Vector{SolverMesh}) -> SolverMesh

Merge multiple SolverMesh objects into a single unified mesh.
Triangles are concatenated in order: mesh 1 first, then mesh 2, etc.
Bounding box is recomputed from the combined geometry.
"""
function merge_meshes(meshes::Vector{SolverMesh})
    if length(meshes) == 1
        return meshes[1]
    end

    total_n = sum(length(m.triangles) for m in meshes)

    all_triangles = Vector{eltype(meshes[1].triangles)}()
    all_normals   = Vector{SVector{3, Float64}}()
    all_areas     = Vector{Float64}()
    all_centers   = Vector{SVector{3, Float64}}()

    sizehint!(all_triangles, total_n)
    sizehint!(all_normals, total_n)
    sizehint!(all_areas, total_n)
    sizehint!(all_centers, total_n)

    for m in meshes
        append!(all_triangles, m.triangles)
        append!(all_normals, m.normals)
        append!(all_areas, m.areas)
        append!(all_centers, m.centers)
    end

    # Recompute global bounding box
    min_x, min_y, min_z =  Inf,  Inf,  Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    for tri in all_triangles
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    @printf("[Geometry] Merged %d meshes → %d total triangles\n", length(meshes), total_n)
    @printf("[Geometry] Combined bounds: [%.3f, %.3f] x [%.3f, %.3f] x [%.3f, %.3f]\n",
            min_x, max_x, min_y, max_y, min_z, max_z)

    return SolverMesh(all_triangles,
                      (min_x, min_y, min_z),
                      (max_x, max_y, max_z),
                      all_normals, all_areas, all_centers)
end

end