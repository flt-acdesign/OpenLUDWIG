# OPEN_Ludwig v1.10 - GPU-Accelerated Lattice Boltzmann CFD Solver

A high-performance, GPU-accelerated Computational Fluid Dynamics (CFD) solver based on the Lattice Boltzmann Method (LBM). Features D3Q27 velocity discretization, cumulant collision operator, WALE turbulence modeling, multi-level adaptive grid refinement, Bouzidi interpolated boundary conditions, optional compressibility correction with thermal DDF, and surface-based aerodynamic force computation.

![Julia](https://img.shields.io/badge/Julia-1.12+-purple.svg)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green.svg)
![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)

---

## Table of Contents

- [Features](#features)
- [Technical Characteristics](#technical-characteristics)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Setting Up Cases](#setting-up-cases)
- [Configuration Reference](#configuration-reference)
- [Post-Processing](#post-processing)
- [Structural Load Mapping](#structural-load-mapping)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Features

- **D3Q27 Lattice**: Full 27-velocity model for improved isotropy and accuracy
- **Cumulant Collision Operator** (default): Multi-relaxation-time operator with independent relaxation of higher-order cumulants, adaptive omega-4 (Geier 2017), and factored Cauchy-Schwarz realizability limiter
- **Regularized BGK Collision** (legacy): Non-equilibrium stress tensor reconstruction with WALE subgrid model
- **WALE Turbulence Model**: Wall-Adapting Local Eddy-viscosity for LES
- **Wall-Modeled LES (WMLES)**: Enables high Reynolds number simulations with equilibrium wall stress model
- **Multi-Level Grid Refinement**: Automatic nested grid generation around geometry with temporal interpolation
- **Bouzidi Boundary Conditions**: Second-order accurate interpolated bounce-back with surface triangle mapping
- **Compressibility Correction**: Optional O(u^4) equilibrium extending the valid Mach number range to ~0.6-0.9
- **Double Distribution Function (DDF)**: Optional thermal energy field (automatically enabled with compressibility correction) for compressible aerodynamics up to Ma ~1.5
- **GPU Acceleration**: CUDA-based computation via KernelAbstractions.jl with automatic CPU fallback
- **Surface Force Computation**: Aerodynamic coefficients (Cd, Cl, Cm) via surface stress integration with pressure/viscous decomposition
- **VTK Output**: ParaView-compatible unstructured mesh export with surface pressure/shear visualization
- **Built-in 3D Post-Processor**: Browser-based Babylon.js visualizer (`postv9.html`) for surface meshes, pressure/Cp/shear fields, and force vector overlays
- **Structural Load Mapping**: Standalone tool (`map_forces_to_nodes.jl`) to transfer triangle-level CFD forces to structural FE nodes preserving force and moment equilibrium
- **Batch Case Execution**: Run multiple cases sequentially with automatic memory cleanup
- **Model Rotation**: Configurable angle of attack (alpha) and sideslip (beta) applied to STL geometry before simulation

---

## Technical Characteristics

### Comparison with State-of-the-Art LBM Solvers

| Feature | OPEN_Ludwig v1.10 | Academic LBM | Commercial CFD |
|---------|-------------------|--------------|----------------|
| **Velocity Set** | D3Q27 (27 velocities) | D3Q19 typical | N/A (FVM/FEM) |
| **Collision Operator** | Cumulant (default) / Regularized BGK | BGK or MRT | Various RANS/LES |
| **Boundary Treatment** | Bouzidi interpolated (2nd order) | Simple bounce-back (1st order) | Body-fitted mesh |
| **Grid Refinement** | Block-structured multi-level | Uniform or octree | Unstructured |
| **Wall Modeling** | Equilibrium stress (log-law) | Often none | Wall functions |
| **Memory Pattern** | A-B (dual buffer) with temporal storage | Various | N/A |
| **Force Computation** | Surface stress integration | Momentum exchange | Surface integration |
| **Thermal / Compressible** | DDF (optional) | Rare | Built-in |

### Key Technical Innovations

1. **Cumulant Collision Operator**: The default collision operator relaxes cumulants (central moments in cumulant space) independently, providing superior stability compared to BGK. Features adaptive omega-4 coupling (Geier 2017) and a factored Cauchy-Schwarz realizability limiter on 2nd-order cumulants.

2. **Sparse Block Storage**: Only active 8x8x8 blocks near geometry are allocated, reducing memory by 60-80% compared to full-domain approaches.

3. **Bouzidi Sparse Implementation**: Q-values and triangle mappings stored only for boundary cells with coordinate lists, minimizing memory overhead while maintaining second-order accuracy.

4. **WALE Turbulence Model**: Unlike Smagorinsky, WALE correctly predicts zero eddy viscosity in pure shear and near walls without ad-hoc damping functions:
   ```
   nu_t = (C_w * delta)^2 * (S^d_ij S^d_ij)^(3/2) / [(S_ij S_ij)^(5/2) + (S^d_ij S^d_ij)^(5/4)]
   ```

5. **Compressibility Correction**: Adds 3rd and 4th-order Hermite terms to the equilibrium distribution, consistently applied across inlet/outlet BCs, sponge zones, positivity limiter, coarse-to-fine interpolation, and regularized BGK collision.

6. **Double Distribution Function (DDF)**: A second set of 27 distributions evolves the temperature field alongside the flow, enabling thermal effects and proper density-temperature coupling for compressible flows.

7. **Multi-Level Time Stepping**: Recursive 2:1 time step ratio between refinement levels with temporal interpolation ensures smooth boundary conditions at grid interfaces.

8. **Surface-Based Force Computation**: Maps LBM flow data to surface pressure and shear stress on STL triangles, then integrates to obtain total forces and moments. Outputs include pressure/viscous decomposition and per-triangle net force CSV files.

9. **Structural Load Mapping**: Post-processing tool distributes triangle-level net forces to structural FE nodes using a minimum-norm pseudoinverse algorithm that preserves both force and moment equilibrium.

### Lattice Boltzmann Fundamentals

The solver implements the lattice Boltzmann equation. With the **cumulant operator** (default):
```
Transform f -> central moments -> cumulants
Relax each cumulant independently toward equilibrium
Transform back to distribution space
Stream: f_i(x + c_i dt, t + dt) = f_i_post_collision
```

With the **regularized BGK operator** (legacy):
```
f_i(x + c_i dt, t + dt) = f_i^eq + (1 - omega) f_i^neq,reg + F_i
```

Where:
- `f_i`: Distribution function for velocity direction i
- `f_i^eq`: Maxwell-Boltzmann equilibrium (O(u^2) standard, or O(u^4) with compressibility correction)
- `omega = 1/tau`: Relaxation rate (related to viscosity: nu = (tau - 0.5) / 3)
- `F_i`: External forcing term (wall model contribution)

### Coordinate Convention

Standard aircraft convention:
- **X**: Streamwise direction (inlet -> outlet) -> Drag (Cd)
- **Y**: Spanwise direction -> Side force (Cs)
- **Z**: Vertical direction -> Lift (Cl)

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- **VRAM**: Minimum 4 GB recommended (scales with mesh size; DDF roughly doubles memory)
- **RAM**: 16 GB+ recommended

### Software
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11, macOS
- **Julia**: Version 1.12 or higher
- **CUDA Toolkit**: Version 11.0+ (for GPU acceleration)

---

## Installation

### Step 1: Install Julia

#### Linux (Ubuntu/Debian)
```bash
# Download Julia 1.12
wget https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.4-linux-x86_64.tar.gz

# Extract
tar -xzf julia-1.12.4-linux-x86_64.tar.gz

# Add to PATH (add to ~/.bashrc for persistence)
export PATH="$PATH:$(pwd)/julia-1.12.4/bin"
```

#### Windows
1. Download the installer from [julialang.org/downloads](https://julialang.org/downloads/)
2. Run the installer and check "Add Julia to PATH"
3. Restart your terminal/PowerShell

#### macOS
```bash
# Using Homebrew
brew install julia

# Or download from julialang.org
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/flt-acdesign/OPEN_Ludwig.git
cd OPEN_Ludwig
```

### Step 3: Install Dependencies

Run the dependency installer script:

```bash
julia src/00_First_time_install_packages.jl
```

This installs:
- `KernelAbstractions` - Backend-agnostic GPU kernels
- `CUDA` - NVIDIA GPU support
- `Adapt` - CPU/GPU data transfer
- `StaticArrays` - High-performance small arrays (SVector)
- `YAML` - Configuration file parsing
- `WriteVTK` - ParaView output
- `Atomix` - Atomic array operations for force integration

**Note**: First-time compilation may take 5-15 minutes.

### Step 4: Verify Installation

```bash
julia -e "using CUDA; CUDA.functional() && println(\"GPU: \", CUDA.name(CUDA.device()))"
```

Expected output: `GPU: NVIDIA GeForce RTX XXXX` (or similar)

---

## Quick Start

### Running Your First Simulation

Note: The configuration files (config.yaml) for each case in this repository should enable comfortable execution on a 4GB VRAM GPU. If you have more VRAM, increase the `surface_resolution` parameter.

1. **Prepare a case** (example: Stanford Bunny is included):
   ```
   CASES/
   └── Stanford_bunny/
       ├── config.yaml    # Configuration file
       └── bunny.stl      # Geometry file
   ```

2. **Edit `cases_to_run.yaml`** to specify which cases to run:
   ```yaml
   case_folders:
     - "Stanford_bunny"
   ```

3. **Run the solver**:
   ```bash
   julia --project=src src/main.jl
   ```

4. **Monitor progress**: The solver prints timestep, physical time, lattice velocity, density, MLUPS (Million Lattice Updates Per Second), and aerodynamic coefficients.

5. **Results** are saved in `CASES/<case_name>/RESULTS/`:
   - `flow_XXXXXX.vtu` - VTK files for flow field visualization
   - `surface_XXXXXX.vtu` - VTK files for surface pressure/shear
   - `surface_XXXXXX.lbmp` - Binary surface files for the built-in post-processor
   - `net_forces_XXXXXX.csv` - Per-triangle net forces (in STL coordinate frame)
   - `convergence.csv` - Time history of solver metrics
   - `forces.csv` - Aerodynamic force/moment history

---

## Setting Up Cases

### Directory Structure

```
CASES/
└── My_New_Case/
    ├── config.yaml      # Required: solver configuration
    └── geometry.stl     # Required: STL geometry file
```

### Creating a New Case

1. **Create a folder** in `CASES/` with your case name
2. **Add your STL file** (ensure it's watertight and in meters or set `stl_scale`)
3. **Create `config.yaml`** (copy from an existing case and modify)
4. **Add case to `cases_to_run.yaml`**

### STL File Guidelines

- **Format**: Binary or ASCII STL
- **Units**: Preferably meters (use `stl_scale` to convert)
- **Quality**: Watertight mesh preferred, but solver will run anyway
- **Orientation**: Flow typically along +X axis
- **Origin**: Position doesn't matter (auto-centered in domain)

---

## Configuration Reference

The `config.yaml` file controls all simulation parameters. Global defaults in `config.yaml` at the project root, per-case overrides in `CASES/<case>/config.yaml`.

### Basic Parameters

```yaml
basic:
  # GEOMETRY
  stl_file: "model.stl"      # STL filename (in case folder)
  stl_scale: 1.0             # Scale factor (0.001 for mm->m)
  alpha: 0.0                 # [deg] Angle of attack (Y-axis rotation)
  beta: 0.0                  # [deg] Sideslip angle (Z-axis rotation)
  minimum_facet_size: 0.0    # [m] Max triangle edge length (0 = no subdivision)

  # MESH RESOLUTION
  surface_resolution: 200    # Cells per reference length (higher = finer mesh)
                             # Memory scales as N^3, compute as N^4
                             # Typical values: 100-300 (coarse), 500-1000 (fine)

  num_levels: 0              # Grid refinement levels
                             # 0 = auto-compute based on domain (recommended)
                             # Each level doubles resolution near geometry

  # REFERENCE VALUES (for coefficient calculation)
  reference_area_of_full_model: 1.0   # [m^2] Planform area for Cd, Cl
  reference_chord: 0.5                # [m] Chord for Cm calculation
  reference_length_for_meshing: 1.0   # [m] Length for Reynolds number
  reference_dimension: "x"            # Which STL dimension is reference length

  # PHYSICS
  fluid:
    density: 1.225                    # [kg/m^3] Air at sea level
    kinematic_viscosity: 1.5e-5       # [m^2/s] Air ~ 1.5e-5, Water ~ 1.0e-6

  flow:
    velocity: 10.0                    # [m/s] Freestream velocity

  # SIMULATION CONTROL
  simulation:
    steps: 10000           # Total timesteps
    ramp_steps: 1500       # Velocity ramp-up period (prevents instability)
    output_freq: 2500      # VTK output every N steps
    output_dir: "RESULTS"  # Output folder name

    output_fields:         # Which fields to save
      density: false
      velocity: true
      velocity_magnitude: true
      vorticity: false
      obstacle: true
      level: true
      bouzidi: false
```

**Understanding `surface_resolution`**:
- This is the most critical parameter for accuracy vs. cost
- Represents cells spanning the reference length at the finest level
- For WMLES, target y+ of 30-100 at the first cell
- Rule of thumb: Start with 200, increase until results converge

**Reynolds Number**: Computed automatically as `Re = velocity * reference_length / kinematic_viscosity`

### Advanced Numerics

```yaml
advanced:
  numerics:
    collision_operator: "cumulant"   # "cumulant" (default) or "regularized_bgk"

    u_lattice: 0.08        # Lattice velocity (Ma_lattice ~ u_lattice/0.577)
                            # Lower = more stable, more accurate
                            # Range: 0.01 (precision) to 0.08 (fast)
                            # Compressibility error is proportional to u_lattice^2

    c_wale: 0.50            # WALE model constant
                            # 0.325 = theoretical, 0.5-0.6 = stable for LBM

    tau_min: 0.501          # Minimum relaxation time (tau > 0.5 required)
                            # Prevents zero/negative viscosity

    tau_safety_factor: 1.0  # Multiplier for tau calculation
                            # >1.0 = more conservative

    inlet_turbulence_intensity: 0.01  # Synthetic inlet fluctuations (0-1)

    # CUMULANT-SPECIFIC (default collision operator)
    cumulant:
      adaptive_omega_4: true    # Geier 2017 automatic 4th-order stabilization
      limiter: "factored"       # "factored" (Cauchy-Schwarz) or "none"

    # STABILITY ENHANCEMENTS
    nu_sgs_background: 0.0          # Minimum SGS viscosity (0.0 for cumulant)
    sponge_blend_distributions: false  # Macroscopic blending is sufficient
    temporal_interpolation: true       # Smooth interface BCs between levels

    # COMPRESSIBILITY (for Ma > 0.3 flows)
    compressibility_correction: false  # Master switch: O(u^4) equilibrium + thermal DDF
                                       # When true, also allocates DDF arrays (~2x VRAM)

  # THERMAL DDF (only active when compressibility_correction: true)
  thermal:
    prandtl: 0.71           # Air Prandtl number
    t_inlet: 1.0            # Inlet temperature (lattice units)
    t_wall: 1.0             # Wall temperature (isothermal BC)
    t_initial: 1.0          # Initial temperature field
    wall_bc: "adiabatic"    # "adiabatic" or "isothermal"
```

**Understanding tau (Tau)**:
- Related to viscosity: `nu = (tau - 0.5) / 3`
- tau -> 0.5 means nu -> 0 (inviscid, unstable)
- With the cumulant operator, tau_min = 0.501 is safe (independent moment relaxation)
- With regularized BGK, use tau_min >= 0.505 and consider nu_sgs_background > 0

**Cumulant vs Regularized BGK**:

| Property | Cumulant (default) | Regularized BGK |
|----------|-------------------|-----------------|
| Stability | Superior (independent moment relaxation) | Good (requires safety margins) |
| nu_sgs_background | 0.0 (not needed) | 0.0005 (recommended) |
| tau_min | 0.501 | 0.505 |
| tau_safety_factor | 1.0 | 1.1 |
| sponge_blend_distributions | false | true |
| ramp_steps | 1500 | 4000 |

### Wall Modeling

```yaml
  high_re:
    auto_levels: false           # Auto-adjust num_levels for memory/stability
    max_levels: 12               # Maximum levels for auto-detection
    min_coarse_blocks: 4         # Minimum blocks in coarsest level

    wall_model:
      enabled: true              # Activate WMLES wall model
      type: "equilibrium"        # Log-law based wall stress
      y_plus_target: 100.0       # Target y+ for first cell

    adaptive_refinement:         # Placeholder for future AMR
      enabled: false
      criterion: "vorticity"
      threshold: 0.1
```

**When to Enable Wall Model**:
- Re > 10^5 with limited resolution
- When direct wall resolution is impractical
- Reduces mesh requirements by 10-100x

### Domain Sizing

```yaml
  domain:
    upstream: 0.75       # Distance upstream (x reference_length)
    downstream: 1.5      # Distance downstream (wake capture)
    lateral: 0.75        # Side clearance
    height: 0.75         # Top/bottom clearance
    sponge_thickness: 0.08   # Boundary absorption layer (fraction, 8% minimum)
```

### Mesh Refinement

```yaml
  refinement:
    block_size: 8               # Cells per block (8 optimal for GPU)
    margin: 2                   # Buffer blocks around geometry
    strategy: "geometry_first"  # Direct STL intersection check

    symmetric_analysis: false   # true = symmetry plane at Y=0
                                # Halves domain, doubles Fx/Fz/My
                                # Automatically forces beta=0

    # Wake refinement zone
    wake_enabled: false
    wake_length: 0.25           # Length behind object (x ref_length)
    wake_width_factor: 0.1
    wake_height_factor: 0.1
```

### Boundary Conditions

```yaml
  boundary:
    method: "bouzidi"       # "bouzidi" (2nd order) or "bounce_back" (1st order)
    bouzidi_levels: 1       # How many fine levels use Bouzidi
    use_float16_qmap: true  # Memory optimization for Q-values
    q_min_threshold: 0.001  # Minimum Q for boundary detection
```

### Force Computation

```yaml
  forces:
    enabled: true              # Compute aerodynamic forces
    output_freq: 0             # 0 = match diagnostics frequency
    moment_center: [0.25, 0.0, 0.0]  # Moment reference point
                                      # [x/chord, y/span, z/thickness]
                                      # 0.25 = quarter-chord (typical)
```

**Force Method**: Surface stress integration
- Maps LBM data (rho, u) to each STL triangle
- Computes pressure: `p = (rho - 1) * cs^2 * pressure_scale`
- Computes wall shear: `tau = mu * (u_tangential / distance)`
- Integrates over surface for total force and moment
- CSV force exports use the unrotated STL coordinate frame (forces inverse-rotated from simulation frame)

### GPU Optimization

```yaml
  gpu:
    async_depth: 8        # Timesteps queued before sync
                          # Higher = faster, less responsive
    use_streams: true     # CUDA stream parallelism
    prefetch_neighbors: true  # Memory prefetching
```

---

## Post-Processing

### Built-in 3D Post-Processor (postv9.html)

LUDWIG includes a browser-based 3D post-processing dashboard at `src/PREPOST/postv9.html`. Open it directly in any modern browser -- no installation or server required.

**Features:**
- **Surface visualization**: Load `.lbmp` binary surface files from `RESULTS/` with per-triangle color mapping
- **Scalar fields**: Switch between Pressure (Pa), Pressure Coefficient (Cp), and Shear Stress (Pa)
- **Force vector overlay**: Load `net_forces_*.csv` or `nodal_loads.csv` as colored 3D arrows
- **Color mapping**: Rainbow colormap (blue-to-red) with manual/auto/symmetric scale controls and color bar
- **Object filtering**: Per-object visibility toggles when the geometry contains multiple named objects
- **Vector slicer**: Axis-aligned clip planes to filter force vectors by X/Y/Z range
- **Camera controls**: Orbit (left-drag), Pan (middle-drag), Right-click to re-center pivot, Shift+drag zoom-to-window
- **Rendering engine**: Babylon.js (WebGL), Z-up convention matching the solver

**Input files:**

| File Type | Format | Source |
|-----------|--------|--------|
| Surface mesh | `.lbmp` (custom binary) | `RESULTS/surface_XXXXXX.lbmp` |
| Net forces | `.csv` | `RESULTS/net_forces_XXXXXX.csv` |
| Nodal loads | `.csv` | Output of `map_forces_to_nodes.jl` |

**Usage:**
1. Open `src/PREPOST/postv9.html` in a browser (Chrome, Firefox, Edge)
2. Click the file picker under "Surface Mesh" and load a `.lbmp` file
3. Optionally load net forces or nodal loads CSV files
4. Use the dropdown to select the scalar field (Pressure, Cp, Shear)
5. Adjust color scale, toggle objects, and explore the geometry

### Viewing Results in ParaView

1. **Open ParaView** (download from [paraview.org](https://www.paraview.org/))

2. **Load VTK files**:
   - Flow field: `File -> Open -> flow_*.vtu`
   - Surface data: `File -> Open -> surface_*.vtu`

3. **Recommended visualizations**:
   - **Velocity magnitude**: Color by "VelocityMagnitude"
   - **Streamlines**: Filters -> Stream Tracer
   - **Iso-surfaces**: Filters -> Contour (Q-criterion for vortices)
   - **Slices**: Filters -> Slice
   - **Surface pressure**: Load surface VTK, color by "Pressure_Pa"
   - **Wall shear**: Color by "ShearMagnitude_Pa"

4. **Animation**: Use the time controls to animate through timesteps

### Surface VTK Fields

The `surface_*.vtu` files contain:
- `Pressure_Pa`: Surface pressure [Pa]
- `ShearX_Pa`, `ShearY_Pa`, `ShearZ_Pa`: Shear stress components [Pa]
- `ShearMagnitude_Pa`: |tau| [Pa]
- `Normal`: Surface normal vectors
- `Area_m2`: Triangle areas [m^2]
- `MappingQuality`: 1.0 if mapped, 0.0 if not

### Analyzing Force Data

The `forces.csv` file contains:

| Column | Description |
|--------|-------------|
| `Step` | Timestep number |
| `Time_phys_s` | Physical time [s] |
| `U_inlet_lat` | Current inlet velocity (lattice units) |
| `Fx_N, Fy_N, Fz_N` | Total forces [N] |
| `Fx_p, Fy_p, Fz_p` | Pressure force components [N] |
| `Fx_v, Fy_v, Fz_v` | Viscous force components [N] |
| `Mx, My, Mz` | Moments [N.m] |
| `Cd, Cl, Cs` | Drag, Lift, Side force coefficients |
| `Cmx, Cmy, Cmz` | Moment coefficients |

### Net Forces CSV

The `net_forces_*.csv` files contain per-triangle force data in the **original STL coordinate frame** (inverse-rotated if alpha/beta are set):

| Column | Description |
|--------|-------------|
| `triangle_id` | Triangle index |
| `object_name` | Object name from STL |
| `cx, cy, cz` | Triangle centroid coordinates [m] |
| `Fx, Fy, Fz` | Net force on triangle [N] |

### Convergence Monitoring

Check `convergence.csv` for:
- `Rho_min`: Should stay above 0.5 (ideally near 1.0)
- `U_inlet_lat`: Current lattice velocity
- `MLUPS`: Performance metric (higher = faster)
- `Cd, Cl`: Force coefficients

**Warning Signs**:
- Rho dropping below 0.5 -> simulation unstable
- Rho oscillating wildly -> reduce `u_lattice` or increase `ramp_steps`
- MLUPS dropping -> memory issues or thermal throttling

---

## Structural Load Mapping

The standalone tool `src/PREPOST/map_forces_to_nodes.jl` maps triangle-level CFD forces (from `net_forces_*.csv`) onto structural finite element nodes, preserving both force and moment equilibrium.

### Usage

```bash
julia src/PREPOST/map_forces_to_nodes.jl <nodes.csv> <net_forces.csv> [output_dir]
```

### Input Files

**Structural nodes CSV** (`nodes.csv`):
```csv
node_id,object_name,cx,cy,cz
1,wing,0.0,0.0,0.0
2,wing,0.5,0.0,0.0
...
```

**Net forces CSV** (`net_forces.csv`): Generated by the solver in `RESULTS/`.

### Output Files

- `nodal_loads.csv` - Per-node force vectors: `node_id, object_name, cx, cy, cz, Fx, Fy, Fz`
- `Nodal_loads_summary.md` - Validation report comparing triangle and nodal resultants

### Algorithm

For each triangle force vector, the tool:
1. Finds the 3 nearest structural nodes in the same object
2. Builds a 6x9 constraint matrix (3 force equilibrium + 3 moment equilibrium equations)
3. Solves for the minimum-norm force distribution via pseudoinverse: `f = pinv(A) * b`
4. Guarantees that the sum of nodal forces equals the original force, and the moment about the centroid is zero

### Dependencies

Julia standard library only (no external packages required).

### Workflow Example

```bash
# 1. Run CFD simulation
julia --project=src src/main.jl

# 2. Map forces to structural nodes
julia src/PREPOST/map_forces_to_nodes.jl \
    my_structure_nodes.csv \
    CASES/Wing_0_deg/RESULTS/net_forces_012000.csv \
    output/

# 3. Visualize in postv9.html
#    Load the surface .lbmp + nodal_loads.csv to see forces on nodes
```

---

## Performance

### Typical Performance (NVIDIA RTX 4090)

| Case Size | Cells | VRAM | MLUPS | Real-time Factor |
|-----------|-------|------|-------|------------------|
| Small (100^3) | 1M | 0.5 GB | 800+ | ~100x |
| Medium (200^3) | 8M | 2 GB | 600 | ~50x |
| Large (400^3) | 64M | 12 GB | 400 | ~20x |
| Very Large (600^3) | 216M | 24 GB | 300 | ~10x |

**MLUPS** = Million Lattice Updates Per Second

### Memory Estimation

Approximate VRAM usage (flow only, with temporal interpolation):
```
VRAM (GB) ~ Total_Cells * 220 bytes / 10^9
```

With DDF enabled (compressibility_correction: true):
```
VRAM (GB) ~ Total_Cells * 440 bytes / 10^9
```

The solver prints a detailed VRAM breakdown after initialization.

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `surface_resolution`
- Reduce `num_levels`
- Enable `use_float16_qmap: true`
- Disable `compressibility_correction` if not needed (halves VRAM for DDF arrays)
- Disable `temporal_interpolation` if not needed

**Simulation explodes (NaN/Inf)**
- Increase `ramp_steps` (try 3000+ for BGK, 1500+ for cumulant)
- Reduce `u_lattice` (try 0.05 or lower)
- Switch to `collision_operator: "cumulant"` (more stable than BGK)
- Enable `adaptive_omega_4: true` and `limiter: "factored"`
- Check STL quality (watertight?)
- Ensure `tau_min` > 0.5

**tau warnings ("tau approaching 0.5")**
- Expected at high Re
- Enable `wall_model` if tau < 0.51 frequently
- With regularized BGK: increase `nu_sgs_background` (try 0.001)
- With cumulant: ensure `adaptive_omega_4: true` (default)

**Slow first run**
- Julia compiles on first execution
- Subsequent runs are much faster
- Use `julia --project=src` to maintain precompilation

**No GPU detected**
- Verify CUDA installation: `nvidia-smi`
- Check Julia CUDA: `julia -e "using CUDA; CUDA.versioninfo()"`
- Ensure GPU drivers are up to date
- The solver automatically falls back to CPU if no GPU is detected

**Poor force coefficient accuracy**
- Increase `surface_resolution`
- Check `reference_area_of_full_model` is correct
- Verify `reference_chord` for moment coefficients
- Ensure simulation has converged (run longer)
- Check surface VTK for `MappingQuality`

**Surface mapping issues**
- Increase force computation `search_radius` parameter
- Check mesh offset in diagnostics output
- Verify STL normals point outward

---

## Code Architecture

```
src/
├── main.jl                          # Entry point, orchestrates simulation
├── dependencies.jl                  # Module includes and constants
├── collision/                       # LBM collision operators
│   ├── kernel_cumulant.jl           # Cumulant D3Q27 collision (default)
│   ├── kernel_regularized_bgk.jl   # Regularized BGK collision (legacy)
│   ├── kernel_BKG_.jl              # Basic BGK kernel
│   ├── kernel_thermal.jl           # Thermal DDF stream-collide kernel
│   ├── lattice.jl                  # D3Q27 lattice definition (weights, velocities)
│   ├── physics_kernels.jl          # GPU kernel orchestration
│   ├── physics_utils.jl            # Equilibrium, gradients, noise
│   ├── physics_interpolation.jl    # Multi-grid interface interpolation
│   └── timestep.jl                 # Timestep execution and kernel dispatch
├── geometry/                        # Mesh and domain
│   ├── geometry.jl                 # STL loading and GPU mesh
│   ├── blocks.jl                   # Block-level data structures (f, f_temp, g, ...)
│   ├── domain.jl                   # Multi-level domain orchestration
│   ├── domain_topology.jl          # Block connectivity management
│   ├── domain_generation.jl        # Voxelization, sponge layers
│   ├── bouzidi_setup.jl            # Bouzidi boundary initialization
│   ├── bouzidi_kernel.jl           # Bouzidi GPU kernel
│   ├── bouzidi_math.jl             # Ray-triangle intersection math
│   └── bouzidi_common.jl           # Shared Bouzidi utilities
├── forces/                          # Aerodynamic force computation
│   ├── structs.jl                  # ForceData structure
│   ├── surface.jl                  # Stress mapping and integration
│   ├── coefficients.jl             # Cd, Cl, Cm calculation
│   ├── global.jl                   # Global force accumulation
│   └── io.jl                       # VTK, CSV, and LBMP output
├── io/                              # Input/output
│   ├── config_loader.jl            # YAML configuration parsing
│   ├── io_vtk.jl                   # Flow field VTK export
│   ├── diagnostics.jl              # Flow statistics and convergence
│   ├── diagnostics_vram.jl         # VRAM usage analysis
│   └── analysis_summary.jl         # Post-run analysis report
├── solver/                          # High-level solver control
│   ├── solver_control.jl           # Recursive multi-level time stepping
│   └── physics_scaling.jl          # Physical to lattice unit conversion
├── PREPOST/                         # Pre/post-processing tools
│   ├── postv9.html                 # Browser-based 3D surface visualizer
│   └── map_forces_to_nodes.jl      # CFD-to-FE structural load mapper
├── Project.toml                     # Julia package manifest
└── Manifest.toml                    # Locked dependency versions
```

---

## Citation

If you use this solver in your research, please cite:

```bibtex
@software{open_ludwig,
  title = {OPEN\_Ludwig: GPU-Accelerated Lattice Boltzmann CFD Solver},
  year = {2026},
  url = {https://github.com/flt-acdesign/OPEN_Ludwig}
}
```

---

## License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE (AGPL-3.0).

---

### Development Notes

- Default collision operator is the cumulant kernel in `collision/kernel_cumulant.jl`
- Force computation pipeline is in `forces/surface.jl`
- Multi-grid stepping logic is in `solver/solver_control.jl`
- All GPU kernels use KernelAbstractions.jl for portability
- Thermal DDF kernel is in `collision/kernel_thermal.jl`
- Post-processor uses Babylon.js for WebGL rendering
