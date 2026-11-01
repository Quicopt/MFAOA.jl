"""
    MFAOA

A solver for binary quadratic programs without constraints (UBQP in the
operations-research literature, QUBO elsewhere), built on the mean-field
approximate optimization algorithm of Misra-Spieldenner, Bode, Schuhmacher,
Stollenwerk, Bagrets and Wilhelm, PRX Quantum 4, 030335 (2023).

The solver minimizes (or maximizes) a quadratic function of binary variables.
It accepts the problem either as a QUBO, `x' * Q * x + c' * x + constant` over
`x ∈ {0,1}ⁿ` ([`QUBOProblem`](@ref)), or in the Ising form the algorithm works
with natively, `h' * σ + Σ_{i<j} J[i,j] σ[i] σ[j]` over `σ ∈ {-1,+1}ⁿ`
([`IsingProblem`](@ref)). Call [`solve`](@ref) with a [`Schedule`](@ref), or
with the number of layers `p` and the time step `τ` of the default linear
schedule. From JuMP, use `Model(MFAOA.Optimizer)` with binary variables and a
quadratic objective.
"""
module MFAOA

import MathOptInterface as MOI

include("kernel.jl")
include("schedule.jl")
include("problem.jl")
include("solve.jl")
include("MOI_wrapper.jl")

export IsingProblem, QUBOProblem, objective, isflipsymmetric
export Schedule, linear_schedule
export SpinState, Workspace, Result, solve, solve!, evolve!, step!

end # module
