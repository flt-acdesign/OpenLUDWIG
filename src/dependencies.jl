using LinearAlgebra
using Printf
using WriteVTK
using Dates
using StaticArrays
using KernelAbstractions
using CUDA
using Adapt
using Base.Threads
using YAML
using Atomix

include("io/config_loader.jl")
include("solver/physics_scaling.jl")

println("[Init] Modules loaded")

include("collision/lattice.jl")

include("geometry/geometry.jl")       # Geometry module (STL loading, mesh structs)
using .Geometry

include("geometry/blocks.jl")         # BlockLevel, SolverMesh, BLOCK_SIZE
include("geometry/bouzidi_common.jl") # BouzidiDataSparse, helpers
include("geometry/bouzidi_math.jl")   # Ray-triangle intersection
include("geometry/bouzidi_setup.jl")  # Q-map computation
include("geometry/bouzidi_kernel.jl") # Bouzidi boundary-condition kernel
include("geometry/domain_topology.jl")# Active blocks, halo, neighbor tables
include("geometry/domain_generation.jl") # Voxelization, flood fill, sponge, wall distances
include("geometry/domain.jl")         # Multi-level domain orchestration

const KAPPA        = 0.41f0
const CS2_PHYSICS  = 1.0f0 / 3.0f0
const CS4_PHYSICS  = CS2_PHYSICS * CS2_PHYSICS

include("collision/physics_utils.jl")          # Equilibrium, gradient helpers, noise
include("collision/physics_interpolation.jl")  # Coarse→fine rescaling interpolation
include("collision/physics_kernels.jl")        # Stream-collide WALE kernel
include("collision/timestep.jl")               # perform_timestep_v2!, build_lattice_arrays_gpu

include("io/diagnostics.jl")
include("io/diagnostics_vram.jl")

include("forces/structs.jl")         # ForceData
include("forces/surface.jl")         # Surface stress mapping & integration
include("forces/io.jl")              # Force CSV / VTK output

include("io/io_vtk.jl")
include("io/analysis_summary.jl")

include("solver/solver_control.jl")