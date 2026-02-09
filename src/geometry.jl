// # FILE: .\src\geometry.jl
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
export SolverMesh
export STLTransform, GeometryPart
export rotate_point, apply_transform_to_triangles
export load_multiple_geometries, merge_geometry_parts, has_dynamic_geometry

"""
    SolverMesh (CPU)
Standard CPU storage for geometry.
"""
struct SolverMesh
    triangles::Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}
    min_bounds::Tuple{Float64, Float64, Float64}
    max_bounds::Tuple{Float64, Float64, Float64}
    
    # CPU Derived properties
    normals::Vector{SVector{3, Float64}}
    areas::Vector{Float64}
    centers::Vector{SVector{3, Float64}}
end

"""
    GPUMesh (GPU)
Flat arrays optimized for GPU kernels.
Removed unused type parameter T.
"""
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

# =============================================================================
# GEOMETRY TRANSFORMATIONS
# =============================================================================

"""
    STLTransform

Defines a spatial transformation for an STL geometry part.
Supports translation, rotation (static angle), and dynamic rotation (angular velocity).

Fields:
- translation: [x, y, z] offset applied after rotation
- rotation_axis: Unit vector defining the rotation axis (e.g., [0,0,1] for Z-axis)
- rotation_angle: Initial rotation angle [radians]
- angular_velocity: Angular velocity [rad/s] - 0 means static geometry
- rotation_center: Point around which rotation occurs (in STL coordinates)
"""
struct STLTransform
    translation::SVector{3, Float64}
    rotation_axis::SVector{3, Float64}
    rotation_angle::Float64
    angular_velocity::Float64
    rotation_center::SVector{3, Float64}
end

STLTransform() = STLTransform(
    SVector(0.0, 0.0, 0.0),
    SVector(0.0, 0.0, 1.0),
    0.0,
    0.0,
    SVector(0.0, 0.0, 0.0)
)

"""
    GeometryPart

Holds an individual STL geometry with its original (untransformed) triangles
and associated transform. Used for multi-STL setups with per-part transformations.
"""
struct GeometryPart
    filename::String
    scale::Float64
    transform::STLTransform
    original_triangles::Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}
end

"""
    rotate_point(point, axis, angle, center)

Rotate a 3D point around an arbitrary axis using Rodrigues' rotation formula.

Arguments:
- point: The 3D point to rotate
- axis: Unit vector defining the rotation axis
- angle: Rotation angle [radians]
- center: Center of rotation
"""
function rotate_point(point::Tuple{Float64,Float64,Float64},
                      axis::SVector{3,Float64}, angle::Float64,
                      center::SVector{3,Float64})
    if abs(angle) < 1e-15
        return point
    end

    # Translate to rotation center
    p = SVector(point[1] - center[1], point[2] - center[2], point[3] - center[3])

    # Rodrigues' rotation formula: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    cosA = cos(angle)
    sinA = sin(angle)
    k = axis

    k_cross_p = cross(k, p)
    k_dot_p = dot(k, p)

    rotated = p * cosA + k_cross_p * sinA + k * k_dot_p * (1.0 - cosA)

    # Translate back
    return (rotated[1] + center[1], rotated[2] + center[2], rotated[3] + center[3])
end

"""
    apply_transform_to_triangles(triangles, transform, time_physical)

Apply a full transformation (rotation + translation) to a set of triangles.
The rotation angle is: initial_angle + angular_velocity * time_physical.

Returns a new vector of transformed triangles.
"""
function apply_transform_to_triangles(
    triangles::Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}},
    transform::STLTransform,
    time_physical::Float64=0.0
)
    total_angle = transform.rotation_angle + transform.angular_velocity * time_physical
    axis = transform.rotation_axis
    center = transform.rotation_center
    trans = transform.translation

    has_rotation = abs(total_angle) > 1e-15
    has_translation = abs(trans[1]) > 1e-15 || abs(trans[2]) > 1e-15 || abs(trans[3]) > 1e-15

    if !has_rotation && !has_translation
        return copy(triangles)
    end

    n = length(triangles)
    result = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}(undef, n)

    @threads for i in 1:n
        v1 = triangles[i][1]
        v2 = triangles[i][2]
        v3 = triangles[i][3]

        # Apply rotation
        if has_rotation
            v1 = rotate_point(v1, axis, total_angle, center)
            v2 = rotate_point(v2, axis, total_angle, center)
            v3 = rotate_point(v3, axis, total_angle, center)
        end

        # Apply translation
        if has_translation
            v1 = (v1[1] + trans[1], v1[2] + trans[2], v1[3] + trans[3])
            v2 = (v2[1] + trans[1], v2[2] + trans[2], v2[3] + trans[3])
            v3 = (v3[1] + trans[1], v3[2] + trans[2], v3[3] + trans[3])
        end

        result[i] = (v1, v2, v3)
    end

    return result
end

"""
    load_multiple_geometries(geometry_configs, case_dir)

Load multiple STL files and store them as GeometryPart objects with their transforms.

Arguments:
- geometry_configs: Vector of Dicts, each with keys: stl_file, stl_scale, translation, rotation
- case_dir: Path to the case directory (for resolving STL file paths)

Returns a Vector{GeometryPart}.
"""
function load_multiple_geometries(geometry_configs::Vector, case_dir::String)
    parts = GeometryPart[]

    for (idx, geo_cfg) in enumerate(geometry_configs)
        stl_file = geo_cfg["stl_file"]
        stl_path = joinpath(case_dir, stl_file)
        scale = Float64(get(geo_cfg, "stl_scale", 1.0))

        if !isfile(stl_path)
            error("[Geometry] STL file not found: $stl_path")
        end

        println("[Geometry] Loading part $idx: $stl_file (scale=$scale)")

        # Load raw triangles (scaled)
        triangles = nothing
        is_binary = true

        open(stl_path, "r") do io
            if filesize(stl_path) < 84
                is_binary = false
            else
                header = String(read(io, 5))
                if startswith(lowercase(header), "solid")
                    seek(io, 80)
                    count = read(io, UInt32)
                    if filesize(stl_path) != 84 + count * 50
                        is_binary = false
                    end
                end
            end
        end

        if is_binary
            open(stl_path, "r") do io
                triangles = parse_binary_stl(io, scale)
            end
        else
            triangles = parse_ascii_stl(stl_path, scale)
        end

        if isempty(triangles)
            error("[Geometry] No triangles loaded from: $stl_file")
        end

        println("[Geometry]   Loaded $(length(triangles)) triangles")

        # Parse transform
        trans_vec = SVector{3,Float64}(Float64.(get(geo_cfg, "translation", [0.0, 0.0, 0.0])))

        rot_cfg = get(geo_cfg, "rotation", Dict())
        rot_axis = SVector{3,Float64}(Float64.(get(rot_cfg, "axis", [0.0, 0.0, 1.0])))
        # Normalize axis
        axis_norm = norm(rot_axis)
        if axis_norm > 1e-10
            rot_axis = rot_axis / axis_norm
        else
            rot_axis = SVector(0.0, 0.0, 1.0)
        end

        rot_angle_deg = Float64(get(rot_cfg, "angle", 0.0))
        rot_angle = deg2rad(rot_angle_deg)
        angular_vel = Float64(get(rot_cfg, "angular_velocity", 0.0))
        rot_center = SVector{3,Float64}(Float64.(get(rot_cfg, "center", [0.0, 0.0, 0.0])))

        transform = STLTransform(trans_vec, rot_axis, rot_angle, angular_vel, rot_center)

        if abs(angular_vel) > 1e-10
            println("[Geometry]   Dynamic rotation: ω=$(angular_vel) rad/s around axis $(Tuple(rot_axis))")
        end
        if abs(rot_angle_deg) > 1e-10
            println("[Geometry]   Initial rotation: $(rot_angle_deg)° around axis $(Tuple(rot_axis))")
        end
        if norm(trans_vec) > 1e-10
            println("[Geometry]   Translation: $(Tuple(trans_vec))")
        end

        push!(parts, GeometryPart(stl_file, scale, transform, triangles))
    end

    return parts
end

"""
    merge_geometry_parts(parts, time_physical)

Apply transforms to all geometry parts and merge them into a single SolverMesh.

Arguments:
- parts: Vector{GeometryPart} with original triangles and transforms
- time_physical: Current simulation time [s] (for angular velocity)

Returns a SolverMesh with all transformed triangles merged.
"""
function merge_geometry_parts(parts::Vector{GeometryPart}, time_physical::Float64=0.0)
    all_triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}()

    for (idx, part) in enumerate(parts)
        transformed = apply_transform_to_triangles(part.original_triangles, part.transform, time_physical)
        append!(all_triangles, transformed)
    end

    if isempty(all_triangles)
        error("[Geometry] No triangles after merging all geometry parts")
    end

    # Compute bounds
    min_x, min_y, min_z = Inf, Inf, Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf

    for tri in all_triangles
        for p in tri
            min_x = min(min_x, p[1]); max_x = max(max_x, p[1])
            min_y = min(min_y, p[2]); max_y = max(max_y, p[2])
            min_z = min(min_z, p[3]); max_z = max(max_z, p[3])
        end
    end

    normals, areas, centers = compute_geometry_properties(all_triangles)

    n_total = length(all_triangles)
    println("[Geometry] Merged mesh: $n_total triangles from $(length(parts)) parts")
    println("[Geometry] Bounds: [$(round(min_x,digits=3)), $(round(max_x,digits=3))] x [$(round(min_y,digits=3)), $(round(max_y,digits=3))] x [$(round(min_z,digits=3)), $(round(max_z,digits=3))]")

    return SolverMesh(all_triangles, (min_x, min_y, min_z), (max_x, max_y, max_z), normals, areas, centers)
end

"""
    has_dynamic_geometry(parts)

Check if any geometry part has a non-zero angular velocity (dynamic rotation).
"""
function has_dynamic_geometry(parts::Vector{GeometryPart})
    return any(abs(p.transform.angular_velocity) > 1e-10 for p in parts)
end

# =============================================================================
# GPU MESH
# =============================================================================

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
    
    # Flatten arrays for GPU
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

function parse_binary_stl(io::IO, scale::Float64)
    skip(io, 80)
    count = read(io, UInt32)
    triangles = Vector{Tuple{Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}, Tuple{Float64,Float64,Float64}}}(undef, count)
    for i in 1:count
        skip(io, 12)
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
        triangles[i] = ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3))
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

function load_mesh(filename::String; scale=1.0)
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

    return SolverMesh(triangles, (min_x, min_y, min_z), (max_x, max_y, max_z), normals, areas, centers)
end

function compute_mesh_bounds(mesh::SolverMesh)
    return (min_bounds=mesh.min_bounds, max_bounds=mesh.max_bounds)
end

end