"""
LATTICE.JL - D3Q27 Lattice Definition
"""

using StaticArrays

const F = Float32

"""
Build the D3Q27 lattice constants as static arrays.
Returns: (cx, cy, cz, weights, opposite, mirror_y)
"""
function build_d3q27_static()
    cx_list = Int32[]
    cy_list = Int32[]
    cz_list = Int32[]
    w_list = F[]
    
    for z in -1:1, y in -1:1, x in -1:1
        push!(cx_list, x)
        push!(cy_list, y)
        push!(cz_list, z)
        
        d2 = x^2 + y^2 + z^2
        if d2 == 0
            push!(w_list, F(8/27))
        elseif d2 == 1
            push!(w_list, F(2/27))
        elseif d2 == 2
            push!(w_list, F(1/54))
        elseif d2 == 3
            push!(w_list, F(1/216))
        end
    end
    
    opp_list = zeros(Int32, 27)
    mirror_y_list = zeros(Int32, 27)

    for i in 1:27
        for j in 1:27
            # Opposite direction
            if cx_list[j] == -cx_list[i] && 
               cy_list[j] == -cy_list[i] && 
               cz_list[j] == -cz_list[i]
                opp_list[i] = j
            end
            # Mirror in Y (for symmetry plane)
            if cx_list[j] == cx_list[i] && 
               cy_list[j] == -cy_list[i] && 
               cz_list[j] == cz_list[i]
                mirror_y_list[i] = j
            end
        end
    end
    
    return (
        SVector{27, Int32}(cx_list), 
        SVector{27, Int32}(cy_list), 
        SVector{27, Int32}(cz_list), 
        SVector{27, F}(w_list), 
        SVector{27, Int32}(opp_list), 
        SVector{27, Int32}(mirror_y_list)
    )
end

# Build lattice constants at module load time
const C_X, C_Y, C_Z, W, OPP, MIRROR_Y = build_d3q27_static()

# Lattice speed of sound squared
const CS2_LATTICE = F(1/3)
const CS4_LATTICE = CS2_LATTICE * CS2_LATTICE
