// # FILE: .\src\bouzidi_math.jl
using StaticArrays
using LinearAlgebra

"""
BOUZIDI_MATH.JL - Geometric Intersection Logic
"""

@inline function ray_triangle_intersection(origin::SVector{3,Float64}, 
                                           dir::SVector{3,Float64},
                                           v1::SVector{3,Float64}, 
                                           v2::SVector{3,Float64}, 
                                           v3::SVector{3,Float64})
    EPSILON = 1e-9
    
    edge1 = v2 - v1
    edge2 = v3 - v1
    h = cross(dir, edge2)
    a = dot(edge1, h)
    
    if abs(a) < EPSILON
        return (false, 0.0)
    end
    
    f = 1.0 / a
    s = origin - v1
    u = f * dot(s, h)
    
    if u < 0.0 || u > 1.0
        return (false, 0.0)
    end
    
    q_vec = cross(s, edge1)
    v = f * dot(dir, q_vec)
    
    if v < 0.0 || u + v > 1.0
        return (false, 0.0)
    end
    
    t = f * dot(edge2, q_vec)
    
    if t > EPSILON
        return (true, t)
    else
        return (false, 0.0)
    end
end

"""
    compute_q_for_cell - Compute Q-values and identify intersecting Triangles
    Returns: (q_values, triangle_indices)
"""
function compute_q_for_cell(cell_center::SVector{3,Float64},
                            dx::Float64,
                            triangles::Vector,
                            mesh_offset::SVector{3,Float64},
                            cx::Vector{Int32},
                            cy::Vector{Int32},
                            cz::Vector{Int32},
                            original_indices::Vector{Int})
    
    q_values = zeros(Float64, 27)
    tri_indices = zeros(Int32, 27)
    
    for k in 1:27
        dir = SVector(Float64(cx[k]), Float64(cy[k]), Float64(cz[k]))
        
        if cx[k] == 0 && cy[k] == 0 && cz[k] == 0
            q_values[k] = 0.0
            continue
        end
        
        dir_norm = dir / norm(dir)
        min_t = Inf
        best_tri_idx = -1
        
        for (i, tri) in enumerate(triangles)
            v1 = SVector{3,Float64}(tri[1]) + mesh_offset
            v2 = SVector{3,Float64}(tri[2]) + mesh_offset
            v3 = SVector{3,Float64}(tri[3]) + mesh_offset
            
            hit, t = ray_triangle_intersection(cell_center, dir_norm, v1, v2, v3)
            
            if hit && t < min_t
                min_t = t
                best_tri_idx = original_indices[i]
            end
        end
        
        if min_t < Inf
            c_magnitude = norm(dir)
            q = min_t / (dx * c_magnitude)
            
            if q > 0.0 && q <= 1.0
                q_values[k] = q
                tri_indices[k] = Int32(best_tri_idx)
            end
        end
    end
    
    return q_values, tri_indices
end