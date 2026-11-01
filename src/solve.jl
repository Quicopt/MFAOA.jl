# solve.jl — running the algorithm on a problem.

"""
    solve!(ws::Workspace, prob::IsingProblem, sched::Schedule) -> ws

Run the algorithm on `prob` with the schedule `sched`, using only the memory
in `ws`. Afterwards `ws.state` holds the final spins, `ws.m` their effective
field and `ws.σ` the rounded solution. Allocates nothing.

Throws an `ArgumentError` when [`isflipsymmetric`](@ref)`(prob)`: the
algorithm cannot break the symmetry itself; fix one variable first.
"""
function solve!(ws::Workspace, prob::IsingProblem, sched::Schedule)
    isflipsymmetric(prob) && throw(ArgumentError(
        "the objective is invariant under flipping all spins (all local fields vanish), " *
        "so the mean-field dynamics never leave the starting point; fix one variable to break the symmetry"))
    length(ws.m) == nvariables(prob) ||
        throw(ArgumentError("workspace is for $(length(ws.m)) variables, problem has $(nvariables(prob))"))
    reset!(ws.state)
    evolve!(ws, prob, sched)
    magnetization!(ws.m, prob.h, prob.J, ws.state.z)
    readout!(ws.σ, ws.state, ws.m, prob)
    ws
end

"""
    Result

What [`solve`](@ref) returns.

- `σ::Vector{Int8}`: the solution as spins, entries ±1.
- `x::Vector{Int8}`: the same solution as binary variables, `x = (1 + σ) / 2`.
- `objective`: the objective value at the solution, in the problem's sense and
  including its constant.
- `objective_mf`: the objective evaluated at the final continuous spins
  (`z` components in place of `σ`). It equals `objective` when every spin has
  reached a pole; a gap between the two means the sweep was too fast.
- `state::SpinState`: the final spins, for inspection.
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

solve(prob::QUBOProblem, sched::Schedule) = solve(IsingProblem(prob), sched)

solve(prob::Union{IsingProblem,QUBOProblem}; p::Integer = 1000, τ::Real = 0.5) =
    solve(prob, linear_schedule(p, τ; T = eltype(prob)))
