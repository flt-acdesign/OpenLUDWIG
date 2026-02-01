// # FILE: .\src\bouzidi.jl
"""
BOUZIDI.JL - Interpolated Bounce-Back Boundary Condition (Wrapper)

This module handles the Bouzidi Interpolated Boundary Condition (IBM).
It is split into four components:
1. `bouzidi_common.jl`: Data structures and utilities.
2. `bouzidi_math.jl`: Low-level geometric intersection logic.
3. `bouzidi_setup.jl`: Pre-processing (Q-map generation).
4. `bouzidi_kernel.jl`: GPU kernels for runtime application.
"""

# Include components in order of dependency
include("bouzidi_common.jl")
include("bouzidi_math.jl")
include("bouzidi_setup.jl")
include("bouzidi_kernel.jl")