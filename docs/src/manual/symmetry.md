# Symmetric problems and fixing variables

## Max-Cut and other symmetric objectives

The largest cut of a triangle separates one vertex from the other two. In
JuMP, MFAOA rejects this problem:

```@example symmetry
using JuMP, MFAOA

edges = [(1, 2), (2, 3), (1, 3)]
model = Model(MFAOA.Optimizer)
@variable(model, x[1:3], Bin)    # x[v]: the set, 0 or 1, that vertex v is in
@objective(model, Max, sum(x[i] + x[j] - 2x[i] * x[j] for (i, j) in edges))
optimize!(model)

println(termination_status(model))
println(raw_status(model))
```

Exchanging the two sets of a cut cuts the same edges, so the objective does
not change when every `x[v]` is replaced by `1 - x[v]`. MFAOA starts every
variable at 1/2 and moves it according to the slope of the objective. The
point where all variables are 1/2 lies halfway between every `x` and its
flipped copy `1 - x`. For an objective that takes the same value at both,
the slope there is zero, so no variable would ever move.

Fixing one variable solves this. Of each pair `x` and `1 - x`, exactly one
has `x[1] = 0`, and both have the same objective value. Fixing `x[1]` to 0
therefore keeps the best objective value:

```@example symmetry
fix(x[1], 0)
optimize!(model)

println(termination_status(model))
println("x:   ", round.(Int, value.(x)))
println("cut: ", objective_value(model))
```

## Objectives in separate parts

The same holds for a group of variables that no term of the objective links
to the other variables: if flipping the group does not change the objective,
one of its variables must be fixed. Max-Cut of a graph in two separate
pieces needs one fixed vertex in each piece:

```@example symmetry
edges = [(1, 2), (3, 4)]
model = Model(MFAOA.Optimizer)
@variable(model, x[1:4], Bin)
@objective(model, Max, sum(x[i] + x[j] - 2x[i] * x[j] for (i, j) in edges))
fix(x[1], 0)
optimize!(model)

println(termination_status(model))
println(raw_status(model))
```

MFAOA finds such groups itself and names a variable to fix in each:

```@example symmetry
fix(x[3], 0)
optimize!(model)

println(termination_status(model))
println("cut: ", objective_value(model))
```

A variable that appears in no term of the objective needs no fixing. Any
value is optimal for it, and MFAOA sets it to 1.

## Fixing variables without JuMP

In the Ising form, an objective made of pair terms `σ[i] σ[j]` alone does
not change when every `σ[i]` changes sign. In this form, the symmetry shows as
`h = 0`, everywhere or on a group of variables that no coupling links to
the others. `isflipsymmetric` tests for it, and `solve` throws an error that
names the variables:

```@example symmetry
J = [0.0 1.0 1.0
     1.0 0.0 1.0
     1.0 1.0 0.0]                 # Max-Cut of the triangle, in Ising form
triangle = IsingProblem(zeros(3), J)
println(isflipsymmetric(triangle))
try
    solve(triangle)
catch err
    println(err.msg)
end
```

To fix `σ[i]` to the value `s`, remove variable `i` from the problem. Its
pair terms with the other variables become linear terms, and its own linear
term becomes a constant:

```@example symmetry
# h, J and offset of the problem with σ[i] fixed to s, in the other variables.
function fix_variable(h, J, offset, i, s)
    others = [j for j in eachindex(h) if j != i]
    h_others = h[others] .+ s .* J[others, i]
    return h_others, J[others, others], offset + s * h[i]
end

h, J_others, offset = fix_variable(zeros(3), J, 0.0, 1, 1)
result = solve(IsingProblem(h, J_others; offset))
println("σ:         ", result.σ, " for the vertices 2 and 3")
println("objective: ", result.objective)
```

The objective here is `σ₁σ₂ + σ₁σ₃ + σ₂σ₃`, which is -1 for every cut that
separates one vertex from the other two.

In the QUBO form, fixing `x[i]` to the value `v` adds `v * (Q[i,j] + Q[j,i])`
to `c[j]` for every other variable `j`, and `v * (Q[i,i] + c[i])` to the
constant.
