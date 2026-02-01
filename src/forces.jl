"""
FORCES.JL - Aerodynamic Force Computation Module

This module computes aerodynamic forces and coefficients by:
1. Mapping LBM flow data (ρ, u) to surface pressure and shear stress
2. Integrating these stresses over the STL surface to get total forces/moments
3. Non-dimensionalizing to get Cd, Cl, Cs, Cm coefficients

Coordinate convention (standard aircraft):
- X: Streamwise direction (inlet to outlet) → Drag (Cd)
- Y: Spanwise direction → Side force (Cs)
- Z: Vertical direction → Lift (Cl)

Usage:
```julia
# Create force data structure
force_data = ForceData(gpu_mesh.n_triangles, backend; 
                       rho_ref=params.rho_physical,
                       u_ref=params.u_physical,
                       area_ref=params.reference_area,
                       chord_ref=params.reference_chord,
                       moment_center=params.moment_center,
                       force_scale=params.force_scale,
                       length_scale=params.length_scale,
                       symmetric=SYMMETRIC_ANALYSIS)

# Compute forces (call each diagnostic step)
compute_aerodynamics!(force_data, finest_level, gpu_mesh, backend, params)

# Access results
println("Cd = ", force_data.Cd)
println("Cl = ", force_data.Cl)

# Save surface VTK for visualization
save_surface_vtk("surface_0001", force_data, cpu_mesh)
```

Submodules:
- structs.jl: ForceData structure definition
- surface.jl: Stress mapping and force integration kernels
- io.jl: VTK and CSV output functions
"""

using KernelAbstractions
using CUDA
using Adapt
using Printf
using LinearAlgebra
using StaticArrays
using WriteVTK
using Atomix

# Load submodules
include("forces/structs.jl")
include("forces/surface.jl")
include("forces/io.jl")

# Note: global.jl (momentum exchange method) has been removed.
# All force computation now uses the surface pressure/shear integration method,
# which correctly accounts for both pressure drag and skin friction.