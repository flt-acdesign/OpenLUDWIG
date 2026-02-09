# ==============================================================================
# COLLISION OPERATOR REGISTRY
# ==============================================================================
# This file includes all available collision operator kernels and provides
# the dispatch function used by the timestep driver.
#
# To add a new collision operator:
#   1. Create a new file: kernel_<name>.jl  (copy an existing one as template)
#   2. Define a kernel function: stream_collide_<name>!(...)
#      with the SAME signature as the existing kernels
#   3. Include the file below
#   4. Add a dispatch entry in `get_collision_kernel`
#   5. Add the operator symbol to VALID_COLLISION_OPERATORS in config_loader.jl
# ==============================================================================

using KernelAbstractions

# --- Include all operator kernels ---
include("kernel_regularized_bgk.jl")
include("kernel_cumulant.jl")
include("kernel_thermal.jl")

# --- Kernel dispatch ---
"""
    get_collision_kernel(operator::Symbol, backend)

Return the compiled kernel function for the given collision operator.
All kernels share an identical call signature so the timestep driver
can use them interchangeably.

Supported operators:
  - :regularized_bgk  — Regularized BGK with WALE LES
  - :cumulant          — Cumulant collision (D3Q27)
"""
function get_collision_kernel(operator::Symbol, backend)
    if operator == :regularized_bgk
        return stream_collide_regularized_bgk!(backend)
    elseif operator == :cumulant
        return stream_collide_cumulant!(backend)
    else
        error("[Collision] Unknown operator: :$operator. " *
              "Valid options: :regularized_bgk, :cumulant")
    end
end