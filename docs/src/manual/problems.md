# Problem forms

Without JuMP, you state a problem directly, as a matrix and vectors, and
call `solve`. MFAOA accepts two forms: the QUBO form, in variables that are
0 or 1, and the Ising form, in variables that are -1 or +1.

## The QUBO form

`QUBOProblem(Q; c, constant, sense)` states the problem

```math
\text{minimize} \quad x^\top Q x + c^\top x + \text{constant}
\qquad \text{over } x \in \{0, 1\}^n .
```

The objective `2x₁x₂ - 3x₂x₃ - x₁ - x₂ + x₃` from the [example on the home page](@ref "Example")
has two pair terms, which go into `Q`, and three linear terms, which go into
`c`:

```@example problems
using MFAOA

Q = [0 2  0
     0 0 -3
     0 0  0]
c = [-1, -1, 1]
qubo = QUBOProblem(Q; c)
result = solve(qubo)

println("x:         ", result.x)
println("objective: ", result.objective)
```

The product `x' * Q * x` counts both `Q[i,j]` and `Q[j,i]` for a pair of
variables. A pair term can therefore go into one triangle of `Q`, as above,
or be split between the two triangles, as in this symmetric `Q`:

```@example problems
Q_symmetric = [0    1    0
               1    0 -1.5
               0 -1.5    0]
objective(QUBOProblem(Q_symmetric; c), [0, 1, 1])
```

Since `x[i]^2 = x[i]` for a binary variable, a diagonal entry `Q[i,i]` acts
like a linear term. To maximize instead, pass `sense = :max`. `objective`
then reports values as stated, without a change of sign.

## The Ising form

`IsingProblem(h, J; offset, sense)` states a problem in variables `σ[i]`
that are -1 or +1:

```math
\text{minimize} \quad h^\top \sigma + \sum_{i<j} J_{ij}\, \sigma_i \sigma_j + \text{offset}
\qquad \text{over } \sigma \in \{-1, +1\}^n .
```

`J` must be symmetric and zero on the diagonal. Each pair appears twice in
`J` but counts once in the sum. Some problems are easier to state in this
form; [Max-Cut](@ref "The same problem without JuMP") is one.

Every QUBO has an Ising form, with `x = (1 + σ) / 2`. `IsingProblem(qubo)`
computes it, and the objective values agree:

```@example problems
ising = IsingProblem(qubo)
println("h:      ", ising.h)
println("J:      ", ising.J)
println("offset: ", ising.offset)
println("objective at σ = (-1, 1, 1): ", objective(ising, [-1, 1, 1]))
```

MFAOA works in the Ising form, and `solve` converts a QUBO first. The result
gives the solution in both forms.

## Matrices

`Q` and `J` can be dense matrices, sparse matrices (`SparseMatrixCSC`) or
other matrix types, such as `Symmetric`. A sparse matrix makes the solve
much faster when most entries are zero; see [Performance](@ref).

A problem uses the matrices and vectors it is given, without copying them,
when their element type is already the problem's. Changing them later
changes the problem, except for its field scale, which is computed when the
problem is built (see [Schedules and the field scale](@ref)). Build a new
problem after such a change; that costs no copy either.

A matrix that does not fit the form is rejected when the problem is built:

```@example problems
try
    IsingProblem([1.0, 0.0], [1.0 0.0; 0.0 0.0])
catch err
    println(err.msg)
end
```

## The result

`solve` returns a `Result` with these fields:

- `x`: the solution in the QUBO form, entries 0 and 1;
- `σ`: the same solution in the Ising form, entries -1 and +1;
- `objective`: its objective value, in the problem's sense and including
  the constant;
- `objective_mf`: the objective at the values just before they are rounded
  to the solution; [Schedules and the field scale](@ref) explains how to
  read it;
- `state`: the final state of the algorithm.
