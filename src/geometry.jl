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