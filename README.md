# MFAOA.jl

MFAOA.jl solves binary quadratic programs without constraints: it minimizes
or maximizes `x' * Q * x + c' * x + constant` over vectors `x` of zeros and
ones. This problem class is known as UBQP (unconstrained binary quadratic
programming) or QUBO; Max-Cut is a well-known example. MFAOA runs a fixed
number of steps, 1000 by default, and returns a solution, without a proof
that no better one exists.

## Example

From JuMP:

```julia
using JuMP, MFAOA

model = Model(MFAOA.Optimizer)
@variable(model, x[1:3], Bin)
@objective(model, Min, 2x[1] * x[2] - 3x[2] * x[3] - x[1] - x[2] + x[3])
optimize!(model)

termination_status(model)   # LOCALLY_SOLVED
value.(x)                   # [0.0, 1.0, 1.0]
objective_value(model)      # -3.0
```

Without JuMP, with the objective as a matrix and a vector:

```julia
using MFAOA

Q = [0 2  0
     0 0 -3
     0 0  0]
c = [-1, -1, 1]
result = solve(QUBOProblem(Q; c))

result.x           # [0, 1, 1]
result.objective   # -3.0
```

## Documentation

The documentation covers the two forms in which a problem can be stated,
the JuMP interface, the settings of the algorithm, objectives such as
Max-Cut's that need a fixed variable, and performance. A tutorial solves
Max-Cut on the Gset benchmark graphs. To build it, with Julia 1.11 or later:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Then open `docs/build/index.html`. The first build downloads the five Gset
graphs that the tutorial uses.

## Reference

MFAOA implements the mean-field approximate optimization algorithm of
A. Misra-Spieldenner, T. Bode, P. K. Schuhmacher, T. Stollenwerk, D. Bagrets
and F. K. Wilhelm, *Mean-field approximate optimization algorithm*,
[PRX Quantum **4**, 030335 (2023)](https://doi.org/10.1103/PRXQuantum.4.030335).
