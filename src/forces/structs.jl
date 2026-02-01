"""
STRUCTS.JL - Force Computation Data Structures

ForceData stores:
- Surface stress maps (pressure, shear) for each triangle
- Integrated forces and moments
- Aerodynamic coefficients
- Reference values for non-dimensionalization
"""

using KernelAbstractions

"""
    ForceData

Container for aerodynamic forces, moments, coefficients, and surface stress maps.

Coordinate convention (standard aircraft):
- X: Streamwise (flow direction) → Drag (Cd)
- Y: Spanwise → Side force (Cs)  
- Z: Vertical → Lift (Cl)

Fields:
- Forces: Fx, Fy, Fz [N]
- Moments: Mx, My, Mz [N·m] about moment_center
- Coefficients: Cd (drag), Cl (lift), Cs (side), Cmx, Cmy, Cmz
- Force decomposition: pressure vs viscous contributions
- Reference values: rho_ref, u_ref, area_ref, chord_ref
- Scaling: force_scale, length_scale (for lattice to physical conversion)
- Surface maps: pressure_map, shear_x/y/z_map [Pa] per triangle
"""
mutable struct ForceData
    # Total forces [N]
    Fx::Float64; Fy::Float64; Fz::Float64
    
    # Total moments [N·m]
    Mx::Float64; My::Float64; Mz::Float64
    
    # Force decomposition [N]
    Fx_pressure::Float64; Fy_pressure::Float64; Fz_pressure::Float64
    Fx_viscous::Float64; Fy_viscous::Float64; Fz_viscous::Float64
    
    # Aerodynamic coefficients (dimensionless)
    Cd::Float64   # Drag coefficient (X direction)
    Cl::Float64   # Lift coefficient (Z direction)
    Cs::Float64   # Side force coefficient (Y direction)
    Cmx::Float64  # Roll moment coefficient
    Cmy::Float64  # Pitch moment coefficient
    Cmz::Float64  # Yaw moment coefficient
    
    # Reference values for coefficient calculation
    rho_ref::Float64      # Reference density [kg/m³]
    u_ref::Float64        # Reference velocity [m/s]
    area_ref::Float64     # Reference area [m²]
    chord_ref::Float64    # Reference chord/length for moments [m]
    moment_center::Tuple{Float64, Float64, Float64}  # Moment reference point [m]
    
    # Unit conversion factors
    force_scale::Float64   # Lattice to physical force conversion
    length_scale::Float64  # Grid spacing dx [m]
    
    # Symmetry flag
    symmetric::Bool
    
    # GPU arrays for surface stress mapping
    pressure_map::AbstractArray   # Pressure [Pa] per triangle
    shear_x_map::AbstractArray    # Shear stress x-component [Pa]
    shear_y_map::AbstractArray    # Shear stress y-component [Pa]
    shear_z_map::AbstractArray    # Shear stress z-component [Pa]
    
    # Diagnostic flag
    diagnostics_printed::Bool
end

"""
    ForceData(n_triangles, backend; kwargs...)

Constructor for ForceData. Allocates GPU arrays for surface stress mapping.

Arguments:
- n_triangles: Number of triangles in the STL mesh
- backend: KernelAbstractions backend (CUDABackend or CPU)

Keyword arguments:
- rho_ref: Reference density [kg/m³] (default: 1.225)
- u_ref: Reference velocity [m/s] (default: 10.0)
- area_ref: Reference area [m²] (default: 1.0)
- chord_ref: Reference chord [m] (default: 1.0)
- moment_center: (x, y, z) point for moment calculation [m]
- force_scale: Conversion from lattice to physical forces
- length_scale: Grid spacing dx [m]
- symmetric: True if simulation uses symmetry plane (Y=0)
"""
function ForceData(n_triangles::Int, backend; 
                   rho_ref=1.225, u_ref=10.0, area_ref=1.0, chord_ref=1.0,
                   moment_center=(0.0, 0.0, 0.0), force_scale=1.0, length_scale=1.0,
                   symmetric=false)
    
    # Allocate GPU arrays for surface stress mapping
    p_map = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sx_map = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sy_map = KernelAbstractions.zeros(backend, Float32, n_triangles)
    sz_map = KernelAbstractions.zeros(backend, Float32, n_triangles)

    return ForceData(
        0.0, 0.0, 0.0,           # Forces
        0.0, 0.0, 0.0,           # Moments
        0.0, 0.0, 0.0,           # Pressure forces
        0.0, 0.0, 0.0,           # Viscous forces
        0.0, 0.0, 0.0,           # Force coefficients
        0.0, 0.0, 0.0,           # Moment coefficients
        rho_ref, u_ref, area_ref, chord_ref, 
        moment_center,
        force_scale, length_scale,
        symmetric,
        p_map, sx_map, sy_map, sz_map,
        false  # diagnostics_printed
    )
end

"""
    reset_forces!(fd::ForceData)

Reset all force and moment values to zero. Call before each integration.
"""
function reset_forces!(fd::ForceData)
    fd.Fx = 0.0; fd.Fy = 0.0; fd.Fz = 0.0
    fd.Mx = 0.0; fd.My = 0.0; fd.Mz = 0.0
    fd.Fx_pressure = 0.0; fd.Fy_pressure = 0.0; fd.Fz_pressure = 0.0
    fd.Fx_viscous = 0.0; fd.Fy_viscous = 0.0; fd.Fz_viscous = 0.0
    fd.Cd = 0.0; fd.Cl = 0.0; fd.Cs = 0.0
    fd.Cmx = 0.0; fd.Cmy = 0.0; fd.Cmz = 0.0
end