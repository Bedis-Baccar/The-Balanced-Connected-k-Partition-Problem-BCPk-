############################################################
#  PROJET BCPk 
############################################################

using JuMP, Gurobi
using Graphs, GraphRecipes, Plots, Colors, Dates
using CSV, DataFrames, Printf, Statistics

# Structure pour compter les coupes générées par les callbacks
mutable struct SharedStats
    connex_cuts::Int
    prop_cuts::Int
    SharedStats() = new(0,0)
end

# =========================================================
# 1- Lecture du graphe pondéré
# =========================================================
function readWeightedGraph_paper(file::String)
    println("Lecture du fichier : $file")
    if !isfile(file)
        error("ERREUR: Fichier '$file' introuvable.")
    end
    data = readlines(file)
    line = split(data[2])
    n = parse(Int, line[1])
    m = parse(Int, line[2])
    E = zeros(Int, n, n)
    W = zeros(Int, n)

    for i in 1:n
        line = split(data[3+i])
        try
            W[i] = parse(Int, line[4])
        catch
            W[i] = 1
        end
    end

    for i in 1:m
        line = split(data[4+n+i])
        orig = parse(Int, line[1]) + 1
        dest = parse(Int, line[2]) + 1
        if 1 <= orig <= n && 1 <= dest <= n
            E[orig,dest] = 1
            E[dest,orig] = 1
        end
    end
    println("Lecture OK : n=$n, m=$m")
    return E, W
end

# =========================================================
# 2- Fonctions de connexité
# =========================================================
function connected_components_of_subset(nodes::Vector{Int}, E::Array{Int,2})
    if isempty(nodes)
        return Vector{Vector{Int}}()
    end
    visited = Set{Int}()
    comps = Vector{Vector{Int}}()
    # adjacency limited to the nodes
    adj = Dict(v => [w for w in nodes if E[v,w] == 1] for v in nodes)
    for v in nodes
        if v in visited; continue; end
        comp = Int[]
        push!(visited, v)
        q = [v]
        head = 1
        while head <= length(q)
            u = q[head]; head += 1
            push!(comp, u)
            for w in adj[u]
                if !(w in visited)
                    push!(visited, w)
                    push!(q, w)
                end
            end
        end
        push!(comps, sort(comp))
    end
    return comps
end

function is_graph_connected(E::Array{Int,2})
    n = size(E,1)
    if n == 0 return true end
    visited = Set([1])
    q = [1]
    head = 1
    while head <= length(q)
        v = q[head]; head += 1
        for u in 1:n
            if E[v,u] == 1 && !(u in visited)
                push!(visited, u); push!(q, u)
            end
        end
    end
    return length(visited) == n
end

# NEW: vérifie si toutes les partitions (dict) sont connexes
function is_partition_connected(E::Array{Int,2}, partitions::Dict)
    for (_, nodes) in partitions
        if isempty(nodes) continue end
        comps = connected_components_of_subset(nodes, E)
        if length(comps) > 1
            return false
        end
    end
    return true
end

# =========================================================
# 3- Affichage graphique avec couleurs distinctes
# =========================================================
function display_graph(E::Array{Int,2}, partitions::Dict;
                       title_str::String = "",
                       savepath::Union{String,Nothing}=nothing)
    n = size(E,1)
    G = SimpleGraph(n)
    for i in 1:n, j in i+1:n
        if E[i,j] == 1
            add_edge!(G,i,j)
        end
    end

    k = max(length(keys(partitions)), 2)
    if k == 2
        palette = [RGB(0.85, 0.1, 0.1), RGB(0.1, 0.2, 0.9)]  # rouge/bleu
    else
        base_colors = [
            RGB(0.85, 0.1, 0.1), RGB(0.1, 0.4, 0.9),
            RGB(0.1, 0.75, 0.3), RGB(1.0, 0.65, 0.1),
            RGB(0.6, 0.1, 0.7), RGB(0.0, 0.7, 0.7),
            RGB(0.9, 0.85, 0.15)
        ]
        palette = vcat(base_colors, distinguishable_colors(max(0, k - length(base_colors)), base_colors))
    end

    vertex_colors = fill(RGB(0.9,0.9,0.9), n)
    for (idx, (key, verts)) in enumerate(sort(collect(pairs(partitions)), by=x->x[1]))
        color = palette[mod1(idx, length(palette))]
        for v in verts
            if 1 <= v <= n
                vertex_colors[v] = color
            end
        end
    end

    plt = graphplot(G; names=1:n, nodecolor=vertex_colors,
                    nodesize=0.45, linealpha=0.6,
                    markerstrokewidth=0.8, markerstrokecolor=:black,
                    legend=false)
    title!(plt, title_str)
    if savepath !== nothing
        savefig(plt, savepath)
        println("🖼️ Graphe sauvegardé dans : $savepath")
    end
    return plt
end

# =========================================================
# 4- Heuristiques Cover / Lifted / Contraction
# =========================================================
function greedy_cover(W::Vector{Int}, budget::Float64)
    idxs = sortperm(W, rev=true)
    S, ssum = Int[], 0
    for i in idxs
        push!(S, i)
        ssum += W[i]
        if ssum > budget
            return sort(S)
        end
    end
    return sort(S)
end

function lifted_cover_heu(S::Vector{Int}, W::Vector{Int}, budget::Float64)
    Sset = Set(S)
    ssum = sum(W[v] for v in S)
    for v in sort(S, by=v->W[v])
        if (ssum - W[v]) > budget
            delete!(Sset, v)
            ssum -= W[v]
        end
    end
    return sort(collect(Sset))
end

function contract_integer_components(E::Array{Int,2}, assigned::Vector{Int})
    n = size(E,1)
    mapping = fill(0, n)
    meta_nodes = Vector{Vector{Int}}()
    id = 0

    labels = unique([assigned[v] for v in 1:n if assigned[v] > 0])
    for ℓ in labels
        nodesℓ = [v for v in 1:n if assigned[v]==ℓ]
        comps = connected_components_of_subset(nodesℓ, E)
        for comp in comps
            id += 1
            push!(meta_nodes, comp)
            for v in comp; mapping[v] = id; end
        end
    end
    for v in 1:n
        if mapping[v] == 0
            id += 1
            push!(meta_nodes, [v])
            mapping[v] = id
        end
    end
    m = length(meta_nodes)
    Enew = zeros(Int, m, m)
    for a in 1:m, b in a+1:m
        connected = any(E[u,v]==1 for u in meta_nodes[a], v in meta_nodes[b])
        if connected; Enew[a,b]=1; Enew[b,a]=1; end
    end
    return Enew, meta_nodes, mapping
end

# =========================================================
# 5- Flow-Based (Q1)
# =========================================================
function solve_BCPk_flow(k::Int, file::String; time_limit=60.0)
        t0 = time()
    E,W = readWeightedGraph_paper(file)
    n = length(W)
    wG = sum(W)
    if wG == 0 wG = 1.0 end

    # Construct augmented directed graph D:
    # nodes 1..n = V, nodes n+1..n+k = sources s_1..s_k
    A_D = Tuple{Int,Int}[]   # list of directed arcs in D
    A_set = Set{Tuple{Int,Int}}()
    # Graph arcs (both directions) for each undirected edge of G
    for u in 1:n
        for v in 1:n
            if E[u,v] == 1
                push!(A_D,(u,v))
                push!(A_set,(u,v))
            end
        end
    end
    # Source arcs: for each source s_i (node n+i) -> every v in V
    for i in 1:k, v in 1:n
        push!(A_D,(n+i, v))
        push!(A_set,(n+i,v))
    end

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "OutputFlag", 0)

    # variables f_a >= 0 and y_a in {0,1} for each arc a in A_D
    @variable(model, f[a in A_D] >= 0)
    @variable(model, y[a in A_D], Bin)

    # Objective: maximize total flow out of s_1 (node n+1)
    @objective(model, Max, sum(f[(n+1, v)] for v in 1:n))

    # (7) order of flows: sum(delta+(s_i)) <= sum(delta+(s_{i+1}))
    for i in 1:k-1
        @constraint(model, sum(f[(n+i, v)] for v in 1:n) <= sum(f[(n+i+1, v)] for v in 1:n))
    end

    # (8) flow balance at each v in V: inflow - outflow == W[v]
    for v in 1:n
        inflow = sum(f[(n+i, v)] for i in 1:k) + sum(f[(u,v)] for u in 1:n if (u,v) in A_set)
        outflow = sum(f[(v,u)] for u in 1:n if (v,u) in A_set)
        @constraint(model, inflow - outflow == W[v])
    end

    # (9) Big-M coupling: f_a <= wG * y_a for all arcs
    for a in A_D
        @constraint(model, f[a] <= wG * y[a])
    end

    # (10) each source sends to at most one v: sum(y[(n+i,v)]) <= 1
    for i in 1:k
        @constraint(model, sum(y[(n+i, v)] for v in 1:n) <= 1)
    end

    # (11) each v in V receives from at most one predecessor (source or other v)
    for v in 1:n
        in_y = sum(y[(n+i,v)] for i in 1:k) + sum(y[(u,v)] for u in 1:n if (u,v) in A_set)
        @constraint(model, in_y <= 1)
    end

    # Solve
    optimize!(model)

    # extract objective
    obj = has_values(model) ? objective_value(model) : -1.0

    # Build partition from y values:
    partitions = Dict(i => Int[] for i in 1:k)
    unassigned = Int[]
    if has_values(model)
        # adj list for arcs with y=1
        adj = Dict(i => Int[] for i in 1:(n+k))
        for a in A_D
            if value(y[a]) > 0.5
                u,v = a
                push!(adj[u], v)
            end
        end

        # for each source s_i, BFS/DFS collecting reachable V nodes
        assigned = zeros(Bool, n)
        for i in 1:k
            s_node = n + i
            q = [v for v in adj[s_node] if 1 <= v <= n]
            visited = Set{Int}(q)
            partitions[i] = copy(q)
            head = 1
            while head <= length(q)
                u = q[head]; head += 1
                for v in adj[u]
                    if 1 <= v <= n && !(v in visited)
                        push!(visited, v); push!(q, v); push!(partitions[i], v)
                    end
                end
            end
            for v in partitions[i]; assigned[v] = true; end
        end

        # remaining nodes not assigned by y-arborescences: attach to neighbor assigned class if possible
        for v in 1:n
            if !assigned[v]
                # try to find neighbor assigned
                found = false
                for u in 1:n
                    if E[v,u]==1 && assigned[u]
                        # pick class of u
                        for i in 1:k
                            if u in partitions[i]
                                push!(partitions[i], v)
                                assigned[v] = true
                                found = true; break
                            end
                        end
                    end
                    if found; break; end
                end
                if !found; push!(unassigned, v); end
            end
        end
    else
        println("Aucune solution trouvée (Q1 flow).")
        return (method="Q1-Flow", instance=basename(file), objective=-1.0, time=round(time()-t0, digits=2), cuts=0, Connexe="Unknown")
    end

    # If leftover unassigned nodes remain, assign them to class 1 (safe fallback)
    if !isempty(unassigned)
        append!(partitions[1], unassigned)
    end

    # plot and save
    png = "plot_Q1_Flow_$(basename(file)).png"
    display_graph(E, partitions; title_str="Q1-Flow Obj=$(round(obj,digits=3))", savepath=png)

    elapsed = round(time() - t0, digits=2)
    conn_part = is_partition_connected(E, partitions) ? "Oui" : "Non"
    return (method="Q1-Flow (Sect4)", instance=basename(file), objective=obj, time=elapsed, cuts=0, Connexe=conn_part)
end

# =========================================================
# Helper: read x-values from callback safely
# returns matrix xvals[n,k] filled with Float64 or throws
# =========================================================
function read_x_from_callback(cb_data, x, n::Int, k::Int; node_value=false)
    xvals = zeros(Float64, n, k)
    for v in 1:n, i in 1:k
        try
            if node_value
                xvals[v,i] = callback_node_value(cb_data, x[v,i])
            else
                xvals[v,i] = callback_value(cb_data, x[v,i])
            end
        catch
            # if something fails, set a safe default (0.0) and continue
            xvals[v,i] = 0.0
        end
    end
    return xvals
end

# =========================================================
# 6- Cut-Based + Améliorations (Q2/Q3) - MAJ Q2+Q3
# =========================================================
function solve_BCPk_cutbased_improved(k::Int, file::String;
                                      time_limit=120.0,
                                      use_cover=false,
                                      use_lifted_cover=false,
                                      use_contraction=false, # use domain propagation (UserCut)
                                      force_connexite=true) # use lazy constraints
    t_start = time()
    E, W = readWeightedGraph_paper(file)
    n, wG = length(W), sum(W)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "OutputFlag", 0)
    if use_contraction
        set_optimizer_attribute(model, "Heuristics", 0.0)
        set_optimizer_attribute(model, "Cuts", 0)
    end

    @variable(model, x[1:n,1:k], Bin)
    @objective(model, Max, sum(W[v]*x[v,1] for v in 1:n))
    for i in 1:k-1
        @constraint(model, sum(W[v]*x[v,i] for v in 1:n) <= sum(W[v]*x[v,i+1] for v in 1:n))
    end
    for v in 1:n; @constraint(model, sum(x[v,i] for i in 1:k) <= 1); end

    # Cover initial cuts (Q3)
    cover_cuts_added = 0
    if use_cover || use_lifted_cover
        for i in 1:k-1
            budget = wG / (k - i + 1)
            S = greedy_cover(W, budget)
            if isempty(S); continue; end
            if use_lifted_cover
                S = lifted_cover_heu(S, W, budget)
            end
            @constraint(model, sum(x[v,i] for v in S) <= max(0, length(S)-1))
            cover_cuts_added += 1
        end
    end

    # Shared stats for callbacks
    stats = SharedStats()

    # --- Q2: Lazy constraints for connectivity (only if requested) ---
    if force_connexite
        function connectivity_callback(cb_data)
            # read integer solution (callback_value on LazyConstraint callback)
            xvals = read_x_from_callback(cb_data, x, n, k; node_value=false)
            for i in 1:k
                V_i = [v for v in 1:n if xvals[v,i] > 0.5]
                if length(V_i) <= 1 continue end
                comps = connected_components_of_subset(V_i, E)
                if length(comps) > 1
                    println("... Q2 (Lazy): Violation de connexité k=$i, composantes: $(length(comps)). Ajout d'une coupe...")
                    # choose a small component C to separate
                    C = sort(comps, by=length)[1]
                    # neighbors N(C)
                    N_C = Set{Int}()
                    for v in C, w in 1:n
                        if E[v,w] == 1 && !(w in C)
                            push!(N_C, w)
                        end
                    end
                    if isempty(N_C) && length(V_i) < n
                        println("Warning: composante $C n'a pas de voisin ext.")
                        continue
                    end
                    N_C_vec = collect(N_C)
                    con = @build_constraint(sum(x[v,i] for v in C) <= (length(C)-1) + sum(x[w,i] for w in N_C_vec))
                    JuMP.MOI.submit(model, JuMP.MOI.LazyConstraint(cb_data), con)
                    stats.connex_cuts += 1
                    break
                end
            end
        end
        JuMP.MOI.set(model, JuMP.MOI.LazyConstraintCallback(), connectivity_callback)
    end

    # --- Q3: Domain propagation / UserCuts (approx) ---
    if use_contraction
        function domain_propagation_callback(cb_data)
            # read node values (could be fractional)
            xvals = nothing
            try
                xvals = read_x_from_callback(cb_data, x, n, k; node_value=true)
            catch err
            # En cas d’erreur, on arrête proprement le callback
                println("Erreur de lecture des x-values dans le callback: ", err)
                return
            end
            # detect fixed variables (approx threshold)
            fixed = zeros(Int, n) # 0 = free, >0 = class fixed
            fixed_by_class = [Int[] for _ in 1:k]
            for v in 1:n
                for i in 1:k
                    if xvals[v,i] > 0.9999
                        fixed[v] = i
                        push!(fixed_by_class[i], v)
                        break
                    end
                end
            end

            for i in 1:k
                if isempty(fixed_by_class[i]) continue end
                V_Gi = [v for v in 1:n if fixed[v]==0 || fixed[v]==i]
                comps_Gi = connected_components_of_subset(V_Gi, E)
                main_nodes = Set{Int}()
                fixed_set = Set(fixed_by_class[i])
                for comp in comps_Gi
                    if !isdisjoint(Set(comp), fixed_set)
                        union!(main_nodes, comp)
                    end
                end
                # any node in V_Gi not in main_nodes cannot connect to fixed set => set x[u,i] = 0
                for u in V_Gi
                    if !(u in main_nodes) && xvals[u,i] > 1e-6
                        # submit user cut x[u,i] <= 0
                        con = @build_constraint(x[u,i] <= 0)
                        JuMP.MOI.submit(model, JuMP.MOI.UserCut(cb_data), con)
                        stats.prop_cuts += 1
                    end
                end
            end
        end
        JuMP.MOI.set(model, JuMP.MOI.UserCutCallback(), domain_propagation_callback)
    end

    # Solve
    optimize!(model)

    obj = has_values(model) ? objective_value(model) : -1.0
    partitions = Dict(i => (has_values(model) ? [v for v in 1:n if value(x[v,i])>0.5] : Int[]) for i in 1:k)
    label = "Q2-Cut"
    if use_cover; label *= "+Cover"; end
    if use_lifted_cover; label *= "+Lifted"; end
    if use_contraction; label *= "+Propag"; end
    if force_connexite; label *= "+Connexe"; end

    pngname = "plot_$(label)_$(basename(file)).png"
    display_graph(E, partitions; title_str="$label - Obj=$(round(obj,digits=2))", savepath=pngname)

    elapsed = round(time() - t_start, digits=2)
    total_cuts = cover_cuts_added + stats.connex_cuts + stats.prop_cuts
    conn_part = is_partition_connected(E, partitions) ? "Oui" : "Non"

    return (method=label, instance=basename(file), objective=obj, time=elapsed,
            cuts=total_cuts, Connexe=conn_part)
end

# =========================================================
# 7- Expérimentations
# =========================================================
function run_experiments_on_instance(file::String, k::Int=2)
    results = []
    println("\n--- Lancement Q1 (Flow) ---")
    push!(results, solve_BCPk_flow(k, file))

    println("\n--- Lancement Q2 (Cut-Base) ---")
    push!(results, solve_BCPk_cutbased_improved(k, file))

    println("\n--- Lancement Q2+Q3 (Cut+Cover) ---")
    push!(results, solve_BCPk_cutbased_improved(k, file; use_cover=true))

    println("\n--- Lancement Q2+Q3 (Cut+Lifted) ---")
    push!(results, solve_BCPk_cutbased_improved(k, file; use_lifted_cover=true))

    println("\n--- Lancement Q2+Q3 (Cut+Propagation) ---")
    push!(results, solve_BCPk_cutbased_improved(k, file; use_contraction=true))

    println("\n--- Lancement Q2 (Cut+Connexe) ---")
    push!(results, solve_BCPk_cutbased_improved(k, file; force_connexite=true))

    println("\n--- Lancement Q2+Q3 (TOUT) ---")
    push!(results, solve_BCPk_cutbased_improved(k, file;
                                                use_lifted_cover=true,
                                                use_contraction=true,
                                                force_connexite=true))

    df = DataFrame(results)
    csvname = "results_$(basename(file))_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS")).csv"
    CSV.write(csvname, df)
    println("\nRésumé des résultats écrit dans : $csvname")
    println(df)
    return df
end

# =========================================================
# 8- Lancement principal
# =========================================================
function run_all()
    base_path = "C:/Users/Baccar/OneDrive - ENSTA/Bureau/prj code/od21/"
    file = joinpath(base_path, "random/rnd_n20/m100/a/rnd_20_100_a_1.in")
    if !isfile(file)
        println("Fichier introuvable : $file")
        return
    end
    run_experiments_on_instance(file, 5)
end

# Execute
run_all()


#Tests:
#"random/rnd_n20/m30/a/rnd_20_30_a_1.in"
#"random/rnd_n20/m30/a/rnd_20_30_a_2.in"
#"random/rnd_n20/m50/a/rnd_20_50_a_1.in"
#"gg_05_05/a/gg_05_05_a_1.in"

#others tests:
#"random/rnd_n20/m50/a/rnd_20_50_a_1.in"
#"gg_120_120/b/gg_120_120_b_1.in"
#"gg_05_05/a/gg_05_05_a_1.in"
# "G_ex_papier.txt"
#"gg_05_10/a/gg_05_10_a_2.in"