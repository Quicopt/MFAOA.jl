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
matrix with a vector, so a run costs `p · n²` operations for `n` variables and
allocates no memory while it runs.

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

## Schedules

The two angle sequences of the algorithm are the schedule. `linear_schedule(p, τ)`
builds the standard one with `p` layers and time step `τ`: the problem weight
grows linearly from `τ/p` to `τ`, the driver weight shrinks from `τ` to `τ/p`.
Any other pair of sequences can be given directly:

```julia
sched = Schedule(β, γ)      # β[k]: driver weight of layer k, γ[k]: problem weight
res = solve(prob, sched)
```

More layers make the sweep slower and the result more stable; `τ ≈ 0.5`
keeps the dynamics regular for typical instances.

## Symmetric problems

If every local field vanishes (`h == 0`), the objective does not change when
all spins are flipped at once, and the algorithm cannot choose between the two
mirror images: the spins never leave their starting point. `solve` throws an
`ArgumentError` in that case; `isflipsymmetric(prob)` tests for it. Fix one
variable to break the symmetry before solving. Max-cut is the common example:
the cut value is invariant under exchanging the two sides.

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

## Status

Version 0.1: the algorithm, the problem types and the schedule. A JuMP
interface (via MathOptInterface) is next.
