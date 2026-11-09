# MFAOA.jl

MFAOA.jl solves binary quadratic programs without constraints: it minimizes
or maximizes

```math
f(x) = x^\top Q x + c^\top x + \text{constant}, \qquad x \in \{0, 1\}^n .
```

This problem class is known as UBQP (unconstrained binary quadratic
programming) or QUBO. Max-Cut is a well-known example. MFAOA runs a fixed
number of steps, 1000 by default, and returns a solution without a proof
that no better one exists.

## Installation

MFAOA.jl needs Julia 1.10 or later. Install it with Julia's package manager:

```julia
using Pkg
Pkg.add("MFAOA")
```

The example below also needs JuMP: `Pkg.add("JuMP")`.

## Example

From JuMP, pass `MFAOA.Optimizer` to the model:

```@example home
using JuMP, MFAOA

model = Model(MFAOA.Optimizer)
@variable(model, x[1:3], Bin)
@objective(model, Min, 2x[1] * x[2] - 3x[2] * x[3] - x[1] - x[2] + x[3])
optimize!(model)

println("status:    ", termination_status(model))
println("solution:  ", round.(Int, value.(x)))
println("objective: ", objective_value(model))
```

Of the eight assignments of `x`, `(0, 1, 1)` has the smallest objective
value, -3. The status `LOCALLY_SOLVED` says that MFAOA
returned a solution, without a proof of optimality.

## Contents

- [Max-Cut on the Gset graphs](@ref): a tutorial on a standard benchmark,
  in JuMP and without it.
- [Problem forms](@ref): stating a problem without JuMP, with 0/1 or ±1
  variables.
- [Solving from JuMP](@ref): what a model may contain, the options and the
  statuses.
- [Schedules and the field scale](@ref): the settings that trade run time
  for the quality of the solution.
- [Symmetric problems and fixing variables](@ref): why some objectives, such
  as Max-Cut's, need a fixed variable.
- [Performance](@ref): sparse and dense matrices, and repeated solves that
  allocate no memory.
- [API reference](@ref): every function and type.

## Reference

MFAOA implements the mean-field approximate optimization algorithm of

> A. Misra-Spieldenner, T. Bode, P. K. Schuhmacher, T. Stollenwerk, D. Bagrets
> and F. K. Wilhelm, *Mean-field approximate optimization algorithm*,
> PRX Quantum **4**, 030335 (2023),
> [doi:10.1103/PRXQuantum.4.030335](https://doi.org/10.1103/PRXQuantum.4.030335).
