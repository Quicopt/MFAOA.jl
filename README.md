# MFAOA.jl

A solver for binary quadratic programs without constraints: minimize or
maximize a quadratic function of variables that are each 0 or 1. The
operations-research literature calls this problem class UBQP (unconstrained
binary quadratic programming); the annealing community calls it QUBO.

The solver implements the mean-field approximate optimization algorithm of
Misra-Spieldenner, Bode, Schuhmacher, Stollenwerk, Bagrets and Wilhelm,
[PRX Quantum **4**, 030335 (2023)](https://doi.org/10.1103/PRXQuantum.4.030335).
Each binary variable becomes a classical unit vector, a spin. All spins start
out pointing the same way, and a sequence of `p` layers rotates them: in every
layer each spin turns about the problem axis by an angle set by the field the
other spins exert on it, then about the driver axis by an angle set by the
schedule. At the end, the sign of each spin's problem-axis component is the
value of its variable. One layer costs one multiplication of the coupling
matrix with a vector: `n²` operations for `n` variables with a dense matrix,
one per nonzero coupling with a sparse one. A run allocates no memory while
it runs.

## Usage

```julia
using MFAOA

# minimize x' * Q * x + c' * x over x[i] ∈ {0, 1}
Q = [ 0.0  2.0  0.0
      0.0  0.0 -3.0
      0.0  0.0  0.0 ]
c = [-1.0, -1.0, 1.0]
prob = QUBOProblem(Q; c = c)

res = solve(prob; p = 1000, τ = 0.5)
res.x          # [0, 1, 1]: the solution as binary variables
res.objective  # -3.0: the objective at the solution
```

Pass `sense = :max` to maximize instead. Objective values are always reported
in the sense you asked for.

The algorithm works natively on the Ising form of the same problem, with
variables `σ[i] = ±1`:

```julia
# minimize h' * σ + Σ_{i<j} J[i,j] σ[i] σ[j]
h = [-0.1, 0.0, 0.0, 0.0]
J = [ 0.0 -1.0  0.0  0.0
     -1.0  0.0 -1.0  0.0
      0.0 -1.0  0.0 -1.0
      0.0  0.0 -1.0  0.0 ]
res = solve(IsingProblem(h, J); p = 500)
res.σ          # [1, 1, 1, 1]: negative couplings favour aligned spins
```

`J` must be symmetric with a zero diagonal. A `QUBOProblem` is converted to
this form by `IsingProblem(prob)`.

Both `Q` and `J` can be sparse matrices from the `SparseArrays` standard
library. Most large problems couple each variable to only a few others, and
then a sparse matrix is much faster. On Max-Cut instance G55 of the Gset
benchmark, 5000 variables and 12 498 couplings, a sparse run takes 0.05 s
and a dense one 3 s.

```julia
using SparseArrays

# 5000 spins on a ring, each coupled to its two neighbours
n = 5000
i = 1:n
j = mod1.(i .+ 1, n)
J = sparse([i; j], [j; i], -1.0, n, n)   # negative couplings favour aligned neighbours
h = fill(-0.25, n)                       # a field favouring σ = +1 everywhere
res = solve(IsingProblem(h, J))
res.objective  # -6250.0: all spins +1, every coupling and every field satisfied
```

The matrices are kept as they are, not copied, so changing them after
building the problem changes the problem.

## Schedules

The two angle sequences of the algorithm are the schedule. `linear_schedule(p, τ)`
builds the standard one with `p` layers and time step `τ`: the problem weight
grows linearly from `τ/p` to `τ`, the driver weight shrinks from `τ` to `τ/p`.
Any other pair of sequences can be given directly:

```julia
sched = Schedule(β, γ)      # β[k]: driver weight of layer k, γ[k]: problem weight
res = solve(prob, sched)
```

More layers make the sweep slower and the result more stable. `τ` between
0.25 and 0.5 works well for typical instances.

The problem weight multiplies the field each spin feels, and that field is
measured relative to its typical size, `fieldscale(prob)`. So the units of
the objective do not matter: multiplying the objective by 1000 gives the
same solution, and one schedule serves problems of different sizes and
coefficient ranges. To measure fields in the units of `h` and `J` instead,
build the problem with `IsingProblem(h, J; scale = 1)`.

## Symmetric problems

In Max-Cut, exchanging the two sets of a cut cuts the same edges: the
objective does not change when every variable is flipped (`σ → -σ`, or
`x → 1 - x`). MFAOA cannot solve such a problem as stated. It starts every
variable halfway between its two values, where such an objective has zero
slope in every variable, so no variable would ever move. The same holds for a
group of variables that no term of the objective links to the others, if
flipping the group leaves the objective unchanged. `isflipsymmetric(prob)`
tests for such groups, and `solve` throws an `ArgumentError` that names them.
Fix one variable of each group to either value before solving. This loses
nothing, since flipping a group does not change the objective value.

## Repeated solves without allocation

`solve` allocates a `Workspace` and returns a `Result`. To solve many problems
of the same size, convert each to Ising form, allocate the workspace once and
use `solve!`, which allocates nothing:

```julia
ising = IsingProblem(prob)
ws = Workspace(ising)
solve!(ws, ising, sched)
ws.σ           # the solution as spins, ±1
```

## Faster dense problems

The product of a dense coupling matrix with a vector is done by BLAS, the
linear algebra library Julia is configured with. A faster BLAS speeds up the
solver without any change to your code. On a Mac, `using AppleAccelerate`
switches to Apple's own library, which made dense problems with 500 to 1000
variables about four times faster on an Apple M4.

## From JuMP

`MFAOA.Optimizer` is a [MathOptInterface](https://jump.dev/MathOptInterface.jl/stable/)
optimizer, so a model written in JuMP can be handed to it directly:

```julia
using JuMP, MFAOA

model = Model(MFAOA.Optimizer)
@variable(model, x[1:3], Bin)
@objective(model, Min, 2x[1]*x[2] - 3x[2]*x[3] - x[1] - x[2] + x[3])
optimize!(model)

termination_status(model)   # LOCALLY_SOLVED
value.(x)                   # [0.0, 1.0, 1.0]
objective_value(model)      # -3.0
```

The model may contain binary variables, a linear or quadratic objective to
minimize or maximize, and fixed variables (`fix(x[1], 1)`), which are
substituted out before the solve. The objective is stored as a sparse matrix
unless at least a quarter of all pairs of variables interact. Any other constraint is rejected when it is
added: the solver handles problems without constraints, and does not
translate constraints into penalty terms.

The termination status `LOCALLY_SOLVED` says what the algorithm delivers: a
solution, without a proof that no better one exists. A status of
`INVALID_MODEL` means that MFAOA cannot solve the model, and
`raw_status(model)` says why and what to do. The common case is an objective
like that of Max-Cut, see above; `fix(x[1], 1)` resolves it.

Options are set as attributes:

```julia
set_attribute(model, "p", 5000)       # layers of the linear schedule (default 1000)
set_attribute(model, "tau", 0.4)      # its time step (default 0.5)
set_attribute(model, "beta", β)       # an explicit schedule, used when both
set_attribute(model, "gamma", γ)      # "beta" and "gamma" are set
```

## Status

Version 0.1: the algorithm, the problem types with dense or sparse
matrices, the schedule, and the JuMP interface.
