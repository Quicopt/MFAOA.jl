# Max-Cut on the Gset graphs

Max-Cut asks for a split of a graph's vertices into two sets, such that the
edges between the two sets carry as much total weight as possible. With one
binary variable per vertex, Max-Cut is a quadratic program in binary
variables without constraints, the problem class that MFAOA.jl solves.

This tutorial solves Max-Cut on graphs from Gset, a collection widely used to
compare Max-Cut methods, and compares the cuts with the best ones published.
You need the packages JuMP and MFAOA, and the standard libraries
SparseArrays and Printf.

## Reading a Gset graph

Yinyu Ye distributes the Gset graphs at
[web.stanford.edu/~yyye/yyye/Gset](https://web.stanford.edu/~yyye/yyye/Gset/).
To follow along, download the graphs G1, G11, G22, G43 and G55 from there
into your working directory. Each graph is a text file. Its first line holds
the number of vertices and the number of edges. Every further line describes
one edge by its two end vertices, numbered from 1, and its weight.

```@example gset
# Read the Gset graph in the file `name`, such as "G1", and return its number
# of vertices and its edges as (i, j, weight) tuples.
function read_gset(name)
    lines = readlines(name)
    n_vertices, n_edges = parse.(Int, split(lines[1]))
    edges = map(lines[2:(n_edges + 1)]) do line
        i, j, weight = parse.(Int, split(line))
        (i, j, weight)
    end
    return n_vertices, edges
end

n_vertices, edges = read_gset("G1")
println("vertices: ", n_vertices)
println("edges:    ", length(edges))
```

## Max-Cut in JuMP

Give each vertex `v` a binary variable `x[v]` that says which of the two sets
it is in.
An edge between vertices `i` and `j` is cut when `x[i]` and `x[j]` differ. The
expression `x[i] + x[j] - 2 x[i] x[j]` is 1 in that case and 0 otherwise, so
the value of the cut is its weighted sum over all edges:

```@example gset
using JuMP, MFAOA

model = Model(MFAOA.Optimizer)
@variable(model, x[1:n_vertices], Bin)    # x[v]: the set, 0 or 1, that vertex v is in
@objective(model, Max,
    sum(weight * (x[i] + x[j] - 2 * x[i] * x[j]) for (i, j, weight) in edges))
optimize!(model)
termination_status(model)
```

The solver rejected the model. `raw_status` says why:

```@example gset
raw_status(model)
```

In Max-Cut, exchanging the two sets cuts the same edges, so replacing every
`x[v]` by `1 - x[v]` does not change the objective. Putting vertex 1 in set 0,
as the message suggests, loses no cut: of a cut and its exchanged copy, one
has vertex 1 in set 0, and both have the same value.

```@example gset
fix(x[1], 0)
optimize!(model)
termination_status(model)
```

`LOCALLY_SOLVED` means that the solver finished and returned a solution,
without a proof that no better one exists. The value of the cut is the
objective value, and `value.(x)` says which set each vertex is in.

```@example gset
best_known_cut = 11624             # Benlic and Hao (2013), Table 2
println("cut found:      ", objective_value(model))
println("best known cut: ", best_known_cut)
println("solve time (s): ", round(solve_time(model); digits = 3))
```

The cut is one edge short of the best cut of G1 that has been published.
[Symmetric problems and fixing variables](@ref) explains when variables must
be fixed and how.

## The same problem without JuMP

MFAOA also accepts a problem directly, as a vector and a matrix. The form it
works in has variables ``\sigma_i`` that are -1 or +1 instead of 0 or 1, and
it minimizes

```math
h^\top \sigma + \sum_{i<j} J_{ij}\, \sigma_i \sigma_j ,
```

where ``h`` is a vector and ``J`` is a symmetric matrix that is zero on the
diagonal. This form is called the Ising form, and `IsingProblem(h, J)`
states a problem in it.

Max-Cut is easy to state in this form. Let ``\sigma_i`` say which set vertex
``i`` is in. The product ``\sigma_i \sigma_j`` is -1 for an edge that is cut and
+1 for one that is not, so ``(1 - \sigma_i \sigma_j)/2`` is 1 if the edge is
cut and 0 otherwise. With ``w_{ij}`` the weight of the edge between ``i`` and
``j``, and ``W`` the total weight of all edges, the value of the cut is

```math
\operatorname{cut}(\sigma)
  = \sum_{\text{edges } ij} w_{ij}\, \frac{1 - \sigma_i \sigma_j}{2}
  = \frac{W - f(\sigma)}{2},
\qquad
f(\sigma) = \sum_{\text{edges } ij} w_{ij}\, \sigma_i \sigma_j .
```

The largest cut therefore has the smallest ``f``, and ``f`` is in Ising
form: ``J_{ij} = w_{ij}`` for every edge, and ``h = 0``.

As in JuMP, one vertex must be fixed. Put vertex 1 in the set
``\sigma_1 = +1``. Each edge from vertex 1 to a vertex ``j`` then contributes
``w_{1j} \sigma_j``, a linear term:

```math
f(\sigma) = \sum_{j \ne 1} w_{1j}\, \sigma_j
  + \sum_{\text{edges } ij,\; i, j \ne 1} w_{ij}\, \sigma_i \sigma_j .
```

The remaining problem is in the vertices 2 to ``n``. Its ``h`` holds the weights
of the edges to vertex 1, and its ``J`` the edges among the remaining
vertices. Most pairs of vertices have no edge between them, so ``J`` is best
stored as a sparse matrix:

```@example gset
using SparseArrays

# Max-Cut of a graph in Ising form, with vertex 1 fixed at σ = +1. Returns h and
# J for vertices 2 to n_vertices, and the total weight W of all edges.
function maxcut_matrices(n_vertices, edges)
    rows    = [i for (i, j, weight) in edges]
    columns = [j for (i, j, weight) in edges]
    weights = [float(weight) for (i, j, weight) in edges]
    # the weight of each edge, stored at (i, j) and at (j, i)
    adjacency = sparse([rows; columns], [columns; rows], [weights; weights],
                       n_vertices, n_vertices)
    others = 2:n_vertices
    h = Vector(adjacency[others, 1])    # weight of the edge to vertex 1, or 0 if there is none
    J = adjacency[others, others]       # the edges among vertices 2 to n_vertices
    return h, J, sum(weights)
end

h, J, total_weight = maxcut_matrices(n_vertices, edges)
result = solve(IsingProblem(h, J))
println("cut found: ", (total_weight - result.objective) / 2)
```

JuMP found the same cut.

## Sparse coupling matrices

MFAOA runs a fixed number of steps, 1000 by default, set by the keyword `p`
of `solve`. Each step multiplies `J` by a vector. For
a dense matrix that takes one multiplication per entry, for a sparse matrix
one per stored entry. Graph G55 shows the difference, because it has many
vertices but few edges:

```@example gset
n_vertices, edges = read_gset("G55")
println("vertices: ", n_vertices)
println("edges:    ", length(edges))
```

Solve it once with `J` stored as a sparse matrix and once with `J` stored as
a dense one:

```@example gset
using Printf

h, J, total_weight = maxcut_matrices(n_vertices, edges)
dense_J = Matrix(J)
sparse_problem = IsingProblem(h, J)
dense_problem = IsingProblem(h, dense_J)

solve(dense_problem; p = 1)    # compile the solver for dense matrices before timing it
dense_time = @elapsed dense_result = solve(dense_problem)
sparse_time = @elapsed sparse_result = solve(sparse_problem)

@printf("                         %12s %10s\n", "dense", "sparse")
@printf("entries of J:            %12d %10d\n", length(dense_J), nnz(J))
@printf("memory for J (MB):       %12.1f %10.2f\n",
        Base.summarysize(dense_J) / 1e6, Base.summarysize(J) / 1e6)
@printf("solve time (s):          %12.3f %10.3f\n", dense_time, sparse_time)
@printf("cut found:               %12d %10d\n",
        (total_weight - dense_result.objective) / 2,
        (total_weight - sparse_result.objective) / 2)
```

Both find the same cut. The sparse matrix stores a thousandth of the
entries, needs 0.44 MB of memory instead of 200 MB, and makes the solve many
times faster.

`MFAOA.Optimizer` makes this choice for you: it stores the objective as a
sparse matrix unless at least a quarter of all pairs of variables appear in
it together. An `IsingProblem` uses the matrix you give it as it is.

## Comparison with the best known cuts

The loop below solves four Gset graphs with the default settings and
compares each cut with the best one published. G1, G22 and G43 are random
graphs: every pair of vertices is joined by an edge with the same
probability. G11 is a grid of 8 by 100 vertices that wraps around in both
directions, like the surface of a doughnut, and its edges have weights +1
and -1.

```@example gset
# Read the Gset graph `name`, solve Max-Cut on it with the default
# settings, and return its number of vertices and the cut found.
function solve_gset(name)
    n_vertices, edges = read_gset(name)
    h, J, total_weight = maxcut_matrices(n_vertices, edges)
    result = solve(IsingProblem(h, J))
    return n_vertices, (total_weight - result.objective) / 2
end

# the best published cuts, from Benlic and Hao (2013), Table 2
best_known_cuts = ["G1" => 11624, "G22" => 13359, "G43" => 6660, "G11" => 564]

@printf("%-5s %9s %8s %11s %7s\n", "graph", "vertices", "cut", "best known", "gap")
for (name, best_known_cut) in best_known_cuts
    vertices, cut = solve_gset(name)
    gap = 100 * (best_known_cut - cut) / best_known_cut    # in percent
    @printf("%-5s %9d %8d %11d %6.2f%%\n", name, vertices, cut, best_known_cut, gap)
end
```

On the three random graphs, the cut found is within 0.7 % of the best known.
The grid G11 has the largest gap, about 2.5 %. The settings that control a
run, `p` and `tau`, are the subject of [Schedules and the field scale](@ref).

## References

- C. Helmberg and F. Rendl, *A spectral bundle method for semidefinite
  programming*, SIAM Journal on Optimization **10**, 673–696 (2000),
  [doi:10.1137/S1052623497328987](https://doi.org/10.1137/S1052623497328987).
- U. Benlic and J.-K. Hao, *Breakout local search for the max-cut problem*,
  Engineering Applications of Artificial Intelligence **26**, 1162–1173 (2013),
  [doi:10.1016/j.engappai.2012.09.001](https://doi.org/10.1016/j.engappai.2012.09.001).
