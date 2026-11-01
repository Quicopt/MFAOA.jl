# solve.jl — running the algorithm on a problem.

"""
    solve!(ws::Workspace, prob::IsingProblem, sched::Schedule) -> ws

Run the algorithm on `prob` with the schedule `sched`, using only the memory
in `ws`. Afterwards `ws.state` holds the final vectors that represent the
variables, `ws.m` the slopes of the objective there, and `ws.σ` the solution.
Allocates nothing.

Throws an `ArgumentError` when [`isflipsymmetric`](@ref)`(prob)`, naming the
variables that the algorithm cannot move, or when `ws` was made for a
different number of variables.
"""
function solve!(ws::Workspace, prob::IsingProblem, sched::Schedule)
    n = nvariables(prob)
    length(ws.m) == n ||
        throw(ArgumentError("the workspace is for $(length(ws.m)) variables, but the problem has $n"))
    _mark_groups!(ws.group, ws.σ, prob.h, prob.J) == 0 ||
        throw(ArgumentError(_flip_message(_flip_groups(prob), n, _ising_name, _ISING_WORDS)))
    reset!(ws.state)
    evolve!(ws, prob, sched)
    magnetization!(ws.m, prob.h, prob.J, ws.state.z)
    readout!(ws.σ, ws.state, ws.m, prob)
    ws
end

"""
    Result

What [`solve`](@ref) returns.

- `σ::Vector{Int8}`: the solution in Ising form, entries -1 and +1.
- `x::Vector{Int8}`: the same solution in QUBO form, `x = (1 + σ) / 2`,
  entries 0 and 1.
- `objective`: the objective value of the solution, in the problem's sense
  and including its constant.
- `objective_mf`: the objective at the values between -1 and +1 that the
  algorithm ends with, before they are rounded to `σ`. It equals `objective`
  when every value has reached -1 or +1. A large gap means that variables
  were still undecided at the end; more layers usually close it.
- `state::SpinState`: the final vectors that represent the variables (see
  [`Schedule`](@ref)).
"""
struct Result{T<:AbstractFloat}
    σ::Vector{Int8}
    x::Vector{Int8}
    objective::T
    objective_mf::T
    state::SpinState{T}
end

function Base.show(io::IO, r::Result{T}) where {T}
    print(io, "Result{$T}: $(length(r.σ)) variables, objective = $(r.objective)")
end

"""
    _objective_mf(prob, ws)

The objective at the continuous spins `z = ws.state.z`, from their field
`m = h + J z` in `ws.m`: `h'z + z'Jz/2 + offset = Σᵢ zᵢ (hᵢ + mᵢ) / 2 + offset`.
"""
function _objective_mf(prob::IsingProblem{T}, ws::Workspace{T}) where {T}
    z = ws.state.z
    E = zero(T)
    @inbounds for i in eachindex(z)
        E += z[i] * (prob.h[i] + ws.m[i])
    end
    E / 2 + prob.offset
end

"""
    solve(prob, sched::Schedule) -> Result
    solve(prob; p = 1000, τ = 0.5) -> Result

Solve the [`IsingProblem`](@ref) or [`QUBOProblem`](@ref) `prob` with the
schedule `sched`, or with the [`linear_schedule`](@ref) of `p` layers and time
step `τ`. The run is deterministic. See [`solve!`](@ref) for the
allocation-free form and [`Result`](@ref) for what comes back.
"""
function solve(prob::IsingProblem{T}, sched::Schedule) where {T}
    ws = Workspace(prob)
    solve!(ws, prob, sched)
    σ = copy(ws.σ)
    x = Int8[(s + 1) >> 1 for s in σ]
    Result{T}(σ, x, objective(prob, σ), _objective_mf(prob, ws), ws.state)
end

function solve(prob::QUBOProblem, sched::Schedule)
    ising = IsingProblem(prob)
    groups = _flip_groups(ising)
    isempty(groups) ||
        throw(ArgumentError(_flip_message(groups, nvariables(prob), _qubo_name, _QUBO_WORDS)))
    solve(ising, sched)
end

solve(prob::Union{IsingProblem,QUBOProblem}; p::Integer = 1000, τ::Real = 0.5) =
    solve(prob, linear_schedule(p, τ; T = eltype(prob)))
