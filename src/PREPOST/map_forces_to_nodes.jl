# ==============================================================================
# MAP NET FORCES TO STRUCTURAL NODES
# ==============================================================================
# Standalone Julia script that maps LUDWIG triangle net forces to a cloud of
# structural FE nodes, preserving both force and moment equilibrium.
#
# Usage:
#   julia map_forces_to_nodes.jl <nodes.csv> <net_forces.csv> [output_dir]
#
# Input files:
#   nodes.csv       — node_id, object_name, cx, cy, cz
#   net_forces.csv  — triangle_id, object_name, cx, cy, cz, Fx, Fy, Fz
#
# Output files (written to output_dir, default = directory of net_forces.csv):
#   nodal_loads.csv          — node_id, object_name, cx, cy, cz, Fx, Fy, Fz
#   Nodal_loads_summary.md   — Validation summary with force/moment comparison
#
# Algorithm:
#   For each triangle force vector, find the 3 nearest structural nodes in the
#   same object. Distribute the force to those 3 nodes such that:
#     (1) Sum of nodal forces = original force       (force equilibrium)
#     (2) Sum of r_i x F_i = 0 about the centroid    (moment equilibrium)
#   This is solved as a minimum-norm problem: f = pinv(A) * b, where A is a
#   6x9 constraint matrix and b = [Fx,Fy,Fz, 0,0,0].
#
# Dependencies: Julia standard library only (Printf, LinearAlgebra)
# ==============================================================================

using Printf
using LinearAlgebra

# ==============================================================================
# CSV PARSING
# ==============================================================================

"""
    parse_nodes_csv(path) -> Vector{NamedTuple}

Parse structural nodes CSV with columns: node_id, object_name, cx, cy, cz
"""
function parse_nodes_csv(path::String)
    nodes = NamedTuple{(:node_id, :object_name, :cx, :cy, :cz),
                       Tuple{Int, String, Float64, Float64, Float64}}[]

    open(path, "r") do f
        header = readline(f)  # skip header
        line_num = 1
        for line in eachline(f)
            line_num += 1
            stripped = strip(line)
            isempty(stripped) && continue

            parts = split(stripped, ',')
            if length(parts) < 5
                @warn "[MapForces] Skipping malformed line $line_num in nodes file ($(length(parts)) columns)"
                continue
            end

            push!(nodes, (
                node_id     = parse(Int, strip(parts[1])),
                object_name = strip(parts[2]),
                cx          = parse(Float64, strip(parts[3])),
                cy          = parse(Float64, strip(parts[4])),
                cz          = parse(Float64, strip(parts[5]))
            ))
        end
    end

    return nodes
end

"""
    parse_forces_csv(path) -> Vector{NamedTuple}

Parse LUDWIG net forces CSV with columns: triangle_id, object_name, cx, cy, cz, Fx, Fy, Fz
"""
function parse_forces_csv(path::String)
    forces = NamedTuple{(:triangle_id, :object_name, :cx, :cy, :cz, :Fx, :Fy, :Fz),
                        Tuple{Int, String, Float64, Float64, Float64, Float64, Float64, Float64}}[]

    open(path, "r") do f
        header = readline(f)  # skip header
        line_num = 1
        for line in eachline(f)
            line_num += 1
            stripped = strip(line)
            isempty(stripped) && continue

            parts = split(stripped, ',')
            if length(parts) < 8
                @warn "[MapForces] Skipping malformed line $line_num in forces file ($(length(parts)) columns)"
                continue
            end

            push!(forces, (
                triangle_id = parse(Int, strip(parts[1])),
                object_name = strip(parts[2]),
                cx          = parse(Float64, strip(parts[3])),
                cy          = parse(Float64, strip(parts[4])),
                cz          = parse(Float64, strip(parts[5])),
                Fx          = parse(Float64, strip(parts[6])),
                Fy          = parse(Float64, strip(parts[7])),
                Fz          = parse(Float64, strip(parts[8]))
            ))
        end
    end

    return forces
end

# ==============================================================================
# GROUPING
# ==============================================================================

"""
    group_by_object(items) -> Dict{String, Vector}

Group a vector of NamedTuples by their :object_name field.
"""
function group_by_object(items)
    groups = Dict{String, Vector{eltype(items)}}()
    for item in items
        name = item.object_name
        if !haskey(groups, name)
            groups[name] = eltype(items)[]
        end
        push!(groups[name], item)
    end
    return groups
end

# ==============================================================================
# NEAREST NEIGHBOR SEARCH
# ==============================================================================

"""
    find_k_nearest(cx, cy, cz, node_coords, k) -> Vector{Int}

Find the k nearest nodes to point (cx, cy, cz) using brute-force Euclidean distance.
Returns indices into node_coords. k is clamped to length(node_coords).
node_coords is a Nx3 Matrix{Float64} where each row is [x, y, z].
"""
function find_k_nearest(cx::Float64, cy::Float64, cz::Float64,
                        node_coords::Matrix{Float64}, k::Int)
    n = size(node_coords, 1)
    k = min(k, n)

    # Compute squared distances
    dists = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        dx = node_coords[i, 1] - cx
        dy = node_coords[i, 2] - cy
        dz = node_coords[i, 3] - cz
        dists[i] = dx*dx + dy*dy + dz*dz
    end

    return partialsortperm(dists, 1:k)
end

# ==============================================================================
# FORCE DISTRIBUTION (EQUILIBRIUM PRESERVING)
# ==============================================================================

"""
    skew(r) -> Matrix{Float64}

3x3 skew-symmetric matrix such that skew(r) * v = r x v.
"""
function skew(r::AbstractVector{Float64})
    return [  0.0   -r[3]   r[2];
             r[3]    0.0   -r[1];
            -r[2]   r[1]    0.0  ]
end

"""
    distribute_force(F, centroid, node_positions) -> Matrix{Float64}

Distribute force vector F (applied at centroid) to k nearest nodes at
node_positions (kx3 matrix), preserving force and moment equilibrium.

Returns a kx3 matrix where each row is the force assigned to that node.

For k nodes, the constraint system is:
  - Rows 1-3: Sum(Fi) = F                      (force equilibrium)
  - Rows 4-6: Sum(ri x Fi) = 0                 (moment equilibrium about centroid)
  => A (6 x 3k) * f (3k x 1) = b (6 x 1)

Solved via minimum-norm: f = pinv(A) * b
"""
function distribute_force(F::Vector{Float64}, centroid::Vector{Float64},
                          node_positions::Matrix{Float64})
    k = size(node_positions, 1)

    # Special case: single node gets all force (moment constraint is trivially 0)
    if k == 1
        return reshape(F, 1, 3)
    end

    # Build constraint matrix A (6 x 3k) and RHS b (6)
    A = zeros(6, 3k)
    b = zeros(6)

    # Force equilibrium: sum of all nodal forces = F
    b[1:3] .= F
    for i in 1:k
        A[1, 3(i-1)+1] = 1.0  # Fx
        A[2, 3(i-1)+2] = 1.0  # Fy
        A[3, 3(i-1)+3] = 1.0  # Fz
    end

    # Moment equilibrium about centroid: sum(ri x Fi) = 0
    # b[4:6] = 0 (already)
    for i in 1:k
        ri = node_positions[i, :] - centroid
        S = skew(ri)  # 3x3: S * Fi = ri x Fi
        A[4:6, 3(i-1)+1:3(i-1)+3] .= S
    end

    # Minimum-norm solution via pseudoinverse
    f_vec = pinv(A) * b  # 3k x 1

    # Reshape to k x 3
    result = zeros(k, 3)
    for i in 1:k
        result[i, 1] = f_vec[3(i-1)+1]
        result[i, 2] = f_vec[3(i-1)+2]
        result[i, 3] = f_vec[3(i-1)+3]
    end

    return result
end

# ==============================================================================
# RESULTANT COMPUTATION
# ==============================================================================

"""
    compute_resultant(positions, forces) -> (F_total, M_total)

Compute total force and moment about the origin from a set of point forces.
positions: Nx3 matrix, forces: Nx3 matrix
Returns (F_total::Vector{3}, M_total::Vector{3})
"""
function compute_resultant(positions::Matrix{Float64}, forces::Matrix{Float64})
    n = size(positions, 1)
    F_total = zeros(3)
    M_total = zeros(3)

    @inbounds for i in 1:n
        fx, fy, fz = forces[i, 1], forces[i, 2], forces[i, 3]
        px, py, pz = positions[i, 1], positions[i, 2], positions[i, 3]

        F_total[1] += fx
        F_total[2] += fy
        F_total[3] += fz

        # M = p x F
        M_total[1] += py * fz - pz * fy
        M_total[2] += pz * fx - px * fz
        M_total[3] += px * fy - py * fx
    end

    return F_total, M_total
end

# ==============================================================================
# OUTPUT WRITERS
# ==============================================================================

"""
    write_nodal_loads_csv(path, nodes, nodal_forces)

Write output CSV: node_id, object_name, cx, cy, cz, Fx, Fy, Fz
nodal_forces is a Dict{Int, Vector{Float64}} mapping node index -> [Fx, Fy, Fz]
"""
function write_nodal_loads_csv(path::String, nodes, nodal_forces::Dict{Int, Vector{Float64}})
    open(path, "w") do f
        println(f, "node_id,object_name,cx,cy,cz,Fx,Fy,Fz")
        for (idx, node) in enumerate(nodes)
            F = get(nodal_forces, idx, [0.0, 0.0, 0.0])
            @printf(f, "%d,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    node.node_id, node.object_name,
                    node.cx, node.cy, node.cz,
                    F[1], F[2], F[3])
        end
    end
end

"""
    write_summary_md(path, object_results, global_result, n_nodes_total, n_forces_total, elapsed)

Write validation summary in Markdown format.
object_results: Dict{String, NamedTuple} with per-object validation data.
global_result: NamedTuple with global validation data.
"""
function write_summary_md(path::String, object_results::Dict, global_result::NamedTuple,
                          n_nodes_total::Int, n_forces_total::Int, elapsed::Float64)
    open(path, "w") do f
        println(f, "# Nodal Loads Mapping Summary")
        println(f, "")
        @printf(f, "**Generated:** %s\n", Dates_now_string())
        @printf(f, "**Elapsed time:** %.3f s\n", elapsed)
        @printf(f, "**Total nodes:** %d\n", n_nodes_total)
        @printf(f, "**Total triangle forces:** %d\n", n_forces_total)
        println(f, "")

        # Per-object tables
        println(f, "## Per-Object Validation")
        println(f, "")

        obj_names = sort(collect(keys(object_results)))
        for obj_name in obj_names
            r = object_results[obj_name]
            println(f, "### Object: `$(obj_name)`")
            println(f, "")
            @printf(f, "- **Nodes:** %d\n", r.n_nodes)
            @printf(f, "- **Triangle forces:** %d\n", r.n_forces)
            println(f, "")
            println(f, "| Quantity | Triangle Forces | Nodal Loads | Difference |")
            println(f, "|----------|---------------:|------------:|-----------:|")
            @printf(f, "| Fx [N]   | %+.6e | %+.6e | %+.6e |\n",
                    r.F_tri[1], r.F_nod[1], r.F_tri[1] - r.F_nod[1])
            @printf(f, "| Fy [N]   | %+.6e | %+.6e | %+.6e |\n",
                    r.F_tri[2], r.F_nod[2], r.F_tri[2] - r.F_nod[2])
            @printf(f, "| Fz [N]   | %+.6e | %+.6e | %+.6e |\n",
                    r.F_tri[3], r.F_nod[3], r.F_tri[3] - r.F_nod[3])
            @printf(f, "| Mx [N.m] | %+.6e | %+.6e | %+.6e |\n",
                    r.M_tri[1], r.M_nod[1], r.M_tri[1] - r.M_nod[1])
            @printf(f, "| My [N.m] | %+.6e | %+.6e | %+.6e |\n",
                    r.M_tri[2], r.M_nod[2], r.M_tri[2] - r.M_nod[2])
            @printf(f, "| Mz [N.m] | %+.6e | %+.6e | %+.6e |\n",
                    r.M_tri[3], r.M_nod[3], r.M_tri[3] - r.M_nod[3])
            println(f, "")
        end

        # Global table
        println(f, "## Global Validation (All Objects)")
        println(f, "")
        r = global_result
        println(f, "| Quantity | Triangle Forces | Nodal Loads | Difference |")
        println(f, "|----------|---------------:|------------:|-----------:|")
        @printf(f, "| Fx [N]   | %+.6e | %+.6e | %+.6e |\n",
                r.F_tri[1], r.F_nod[1], r.F_tri[1] - r.F_nod[1])
        @printf(f, "| Fy [N]   | %+.6e | %+.6e | %+.6e |\n",
                r.F_tri[2], r.F_nod[2], r.F_tri[2] - r.F_nod[2])
        @printf(f, "| Fz [N]   | %+.6e | %+.6e | %+.6e |\n",
                r.F_tri[3], r.F_nod[3], r.F_tri[3] - r.F_nod[3])
        @printf(f, "| Mx [N.m] | %+.6e | %+.6e | %+.6e |\n",
                r.M_tri[1], r.M_nod[1], r.M_tri[1] - r.M_nod[1])
        @printf(f, "| My [N.m] | %+.6e | %+.6e | %+.6e |\n",
                r.M_tri[2], r.M_nod[2], r.M_tri[2] - r.M_nod[2])
        @printf(f, "| Mz [N.m] | %+.6e | %+.6e | %+.6e |\n",
                r.M_tri[3], r.M_nod[3], r.M_tri[3] - r.M_nod[3])
        println(f, "")

        # Max absolute differences
        dF = abs.([r.F_tri[i] - r.F_nod[i] for i in 1:3])
        dM = abs.([r.M_tri[i] - r.M_nod[i] for i in 1:3])
        @printf(f, "**Max force difference:**  %.2e N\n", maximum(dF))
        @printf(f, "**Max moment difference:** %.2e N.m\n", maximum(dM))
        println(f, "")

        if maximum(dF) < 1e-6 && maximum(dM) < 1e-6
            println(f, "> **PASS** - Force and moment equilibrium preserved within machine precision.")
        else
            println(f, "> **WARNING** - Differences exceed expected machine precision. Check input data.")
        end
    end
end

"""Simple timestamp string without importing Dates."""
function Dates_now_string()
    # Use Libc.strftime for a timestamp without requiring the Dates package
    return Libc.strftime("%Y-%m-%d %H:%M:%S", time())
end

# ==============================================================================
# MAIN
# ==============================================================================

function main()
    t_start = time()

    println("\n" * "="^70)
    println("[MapForces] Net Force to Structural Node Mapper")
    println("="^70)

    # --- Argument parsing ---
    if length(ARGS) < 2
        println("""
Usage:
  julia map_forces_to_nodes.jl <nodes.csv> <net_forces.csv> [output_dir]

Arguments:
  nodes.csv       Structural nodes: node_id, object_name, cx, cy, cz
  net_forces.csv  LUDWIG net forces: triangle_id, object_name, cx, cy, cz, Fx, Fy, Fz
  output_dir      Output directory (default: same as net_forces.csv)
""")
        return
    end

    nodes_path  = ARGS[1]
    forces_path = ARGS[2]
    output_dir  = length(ARGS) >= 3 ? ARGS[3] : dirname(forces_path)

    if isempty(output_dir)
        output_dir = "."
    end

    if !isfile(nodes_path)
        error("[MapForces] Nodes file not found: $nodes_path")
    end
    if !isfile(forces_path)
        error("[MapForces] Forces file not found: $forces_path")
    end

    # --- Parse input files ---
    println("\n[MapForces] Reading nodes:  $nodes_path")
    nodes = parse_nodes_csv(nodes_path)
    println("[MapForces]   $(length(nodes)) nodes loaded")

    println("[MapForces] Reading forces: $forces_path")
    forces = parse_forces_csv(forces_path)
    println("[MapForces]   $(length(forces)) triangle forces loaded")

    if isempty(nodes) || isempty(forces)
        error("[MapForces] Empty input data. Cannot proceed.")
    end

    # --- Group by object ---
    node_groups  = group_by_object(nodes)
    force_groups = group_by_object(forces)

    all_objects = sort(collect(union(keys(node_groups), keys(force_groups))))
    println("\n[MapForces] Objects found: $(join(all_objects, ", "))")

    # --- Build node index for global accumulation ---
    # nodal_forces[i] = accumulated [Fx, Fy, Fz] for nodes[i]
    nodal_forces = Dict{Int, Vector{Float64}}()

    # --- Process each object ---
    object_results = Dict{String, NamedTuple}()

    for obj_name in all_objects
        println("\n--- Object: $(obj_name) ---")

        obj_forces = get(force_groups, obj_name, nothing)
        obj_nodes  = get(node_groups, obj_name, nothing)

        if obj_forces === nothing || isempty(obj_forces)
            println("  [SKIP] No triangle forces for this object")
            continue
        end
        if obj_nodes === nothing || isempty(obj_nodes)
            @warn "[MapForces] Object '$obj_name' has $(length(obj_forces)) forces but NO nodes — skipping"
            continue
        end

        n_obj_forces = length(obj_forces)
        n_obj_nodes  = length(obj_nodes)
        println("  Nodes: $n_obj_nodes, Triangle forces: $n_obj_forces")

        # Build coordinate matrix for this object's nodes (Nx3)
        node_coords = zeros(n_obj_nodes, 3)
        for (i, nd) in enumerate(obj_nodes)
            node_coords[i, 1] = nd.cx
            node_coords[i, 2] = nd.cy
            node_coords[i, 3] = nd.cz
        end

        # Find global indices for this object's nodes
        # Build a mapping: obj_nodes[local_idx] -> global index in `nodes`
        obj_node_global_idx = zeros(Int, n_obj_nodes)
        for (local_i, nd) in enumerate(obj_nodes)
            for (global_i, gnd) in enumerate(nodes)
                if gnd.node_id == nd.node_id && gnd.object_name == nd.object_name
                    obj_node_global_idx[local_i] = global_i
                    break
                end
            end
        end

        # --- Distribute forces ---
        k_neighbors = min(3, n_obj_nodes)

        for tf in obj_forces
            centroid = [tf.cx, tf.cy, tf.cz]
            F = [tf.Fx, tf.Fy, tf.Fz]

            # Find k nearest nodes
            nearest_local = find_k_nearest(tf.cx, tf.cy, tf.cz, node_coords, k_neighbors)

            # Get positions of nearest nodes (kx3)
            nearest_positions = node_coords[nearest_local, :]

            # Distribute force preserving equilibrium
            nodal_F = distribute_force(F, centroid, nearest_positions)

            # Accumulate on global node indices
            for (j, local_idx) in enumerate(nearest_local)
                global_idx = obj_node_global_idx[local_idx]
                if !haskey(nodal_forces, global_idx)
                    nodal_forces[global_idx] = [0.0, 0.0, 0.0]
                end
                nodal_forces[global_idx][1] += nodal_F[j, 1]
                nodal_forces[global_idx][2] += nodal_F[j, 2]
                nodal_forces[global_idx][3] += nodal_F[j, 3]
            end
        end

        # --- Per-object validation ---
        # Triangle resultant
        tri_positions = zeros(n_obj_forces, 3)
        tri_forces_mat = zeros(n_obj_forces, 3)
        for (i, tf) in enumerate(obj_forces)
            tri_positions[i, :] .= [tf.cx, tf.cy, tf.cz]
            tri_forces_mat[i, :] .= [tf.Fx, tf.Fy, tf.Fz]
        end
        F_tri, M_tri = compute_resultant(tri_positions, tri_forces_mat)

        # Nodal resultant (only nodes belonging to this object)
        nod_positions = zeros(n_obj_nodes, 3)
        nod_forces_mat = zeros(n_obj_nodes, 3)
        for (local_i, nd) in enumerate(obj_nodes)
            global_idx = obj_node_global_idx[local_i]
            nod_positions[local_i, :] .= [nd.cx, nd.cy, nd.cz]
            F = get(nodal_forces, global_idx, [0.0, 0.0, 0.0])
            nod_forces_mat[local_i, :] .= F
        end
        F_nod, M_nod = compute_resultant(nod_positions, nod_forces_mat)

        dF = maximum(abs.(F_tri - F_nod))
        dM = maximum(abs.(M_tri - M_nod))

        @printf("  Triangle resultant: Fx=%+.4e  Fy=%+.4e  Fz=%+.4e\n",
                F_tri[1], F_tri[2], F_tri[3])
        @printf("  Nodal resultant:    Fx=%+.4e  Fy=%+.4e  Fz=%+.4e\n",
                F_nod[1], F_nod[2], F_nod[3])
        @printf("  Max |dF|=%.2e  Max |dM|=%.2e  %s\n",
                dF, dM, (dF < 1e-6 && dM < 1e-6) ? "OK" : "CHECK")

        object_results[obj_name] = (
            n_nodes  = n_obj_nodes,
            n_forces = n_obj_forces,
            F_tri    = F_tri,
            M_tri    = M_tri,
            F_nod    = F_nod,
            M_nod    = M_nod
        )
    end

    # --- Global validation ---
    println("\n" * "-"^40)
    println("[MapForces] Global Validation")

    # Total triangle resultant
    n_all = length(forces)
    all_tri_pos = zeros(n_all, 3)
    all_tri_frc = zeros(n_all, 3)
    for (i, tf) in enumerate(forces)
        all_tri_pos[i, :] .= [tf.cx, tf.cy, tf.cz]
        all_tri_frc[i, :] .= [tf.Fx, tf.Fy, tf.Fz]
    end
    F_tri_global, M_tri_global = compute_resultant(all_tri_pos, all_tri_frc)

    # Total nodal resultant
    n_nodes_with_load = length(nodal_forces)
    nod_pos_all = zeros(max(n_nodes_with_load, 1), 3)
    nod_frc_all = zeros(max(n_nodes_with_load, 1), 3)
    for (count, (global_idx, F)) in enumerate(nodal_forces)
        nd = nodes[global_idx]
        nod_pos_all[count, :] .= [nd.cx, nd.cy, nd.cz]
        nod_frc_all[count, :] .= F
    end
    F_nod_global, M_nod_global = compute_resultant(
        nod_pos_all[1:n_nodes_with_load, :],
        nod_frc_all[1:n_nodes_with_load, :])

    dF_g = maximum(abs.(F_tri_global - F_nod_global))
    dM_g = maximum(abs.(M_tri_global - M_nod_global))

    @printf("  Triangle: F=[%+.4e, %+.4e, %+.4e]  M=[%+.4e, %+.4e, %+.4e]\n",
            F_tri_global..., M_tri_global...)
    @printf("  Nodal:    F=[%+.4e, %+.4e, %+.4e]  M=[%+.4e, %+.4e, %+.4e]\n",
            F_nod_global..., M_nod_global...)
    @printf("  Max |dF|=%.2e  Max |dM|=%.2e  %s\n",
            dF_g, dM_g, (dF_g < 1e-6 && dM_g < 1e-6) ? "PASS" : "CHECK")

    global_result = (
        F_tri = F_tri_global,
        M_tri = M_tri_global,
        F_nod = F_nod_global,
        M_nod = M_nod_global
    )

    # --- Write outputs ---
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    csv_path = joinpath(output_dir, "nodal_loads.csv")
    md_path  = joinpath(output_dir, "Nodal_loads_summary.md")

    elapsed = time() - t_start

    write_nodal_loads_csv(csv_path, nodes, nodal_forces)
    println("\n[MapForces] Written: $csv_path")

    write_summary_md(md_path, object_results, global_result,
                     length(nodes), length(forces), elapsed)
    println("[MapForces] Written: $md_path")

    @printf("\n[MapForces] Done in %.3f s\n", elapsed)
    println("="^70)
end

# ==============================================================================
# ENTRY POINT
# ==============================================================================
main()
