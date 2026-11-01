# schedule.jl — the two angle sequences that drive the algorithm.

"""
    Schedule(β, γ)

The angles that drive the algorithm, one pair `β[k]`, `γ[k]` per layer `k`.

The algorithm represents each variable `σ[i]` by a unit vector in three
dimensions. The vector's z component, between -1 and +1, is the variable's
current value. Every vector starts at `(1, 0, 0)`, so every variable starts
at 0, halfway between -1 and +1. Each layer `k` turns every vector `i` twice:

1. about the z axis by the angle `2γ[k] m[i] / s`, where `m[i]` is the slope
   of the objective in variable `i` at the current values, and `s` is the
   problem's [`fieldscale`](@ref) (the opposite angle for a maximization);
2. about the x axis by the angle `2β[k] Δ[i]`, where `Δ[i]` is 1 unless the
   problem sets it.

After the last layer, `σ[i]` is the sign of the z component. The two vectors
`β` and `γ` must have the same length, the number of layers `p`.
[`linear_schedule`](@ref) builds the usual choice; any other pair of vectors
can be passed directly.
"""
struct Schedule{T<:AbstractFloat}
    β::Vector{T}
    γ::Vector{T}
    function Schedule{T}(β::Vector{T}, γ::Vector{T}) where {T<:AbstractFloat}
        length(β) == length(γ) ||
            throw(ArgumentError("β and γ must have one entry per layer; got $(length(β)) and $(length(γ))"))
        isempty(β) && throw(ArgumentError("a schedule needs at least one layer"))
        new{T}(β, γ)
    end
end

"""
    Schedule(β::AbstractVector, γ::AbstractVector)

Build a schedule from any two real vectors, promoted to a common
floating-point type.
"""
function Schedule(β::AbstractVector, γ::AbstractVector)
    T = float(promote_type(eltype(β), eltype(γ)))
    Schedule{T}(Vector{T}(β), Vector{T}(γ))
end

"""
    length(s::Schedule)

The number of layers `p`.
"""
Base.length(s::Schedule) = length(s.β)

"""
    linear_schedule(p, τ = 0.5; T = Float64)

The usual schedule: over `p` layers, the angles `γ` grow and the angles `β`
shrink linearly,

    γ[k] = τ k / p,    β[k] = τ (1 - (k - 1) / p),    k = 1, …, p.

This is Eq. 13 of Misra-Spieldenner et al., PRX Quantum 4, 030335 (2023).
`τ` sets the size of the angles. Because `γ` is measured in units of the
problem's [`fieldscale`](@ref), the same `τ` suits problems of any size and
units; values between 0.25 and 0.5 work well on typical instances. With more
layers, the angles change more slowly from one layer to the next. That
usually gives a better solution, and the time of a solve grows in proportion
to `p`.
"""
function linear_schedule(p::Integer, τ::Real = 0.5; T::Type{<:AbstractFloat} = Float64)
    p ≥ 1 || throw(ArgumentError("need at least one layer, got p = $p"))
    τT = T(τ)
    β = [τT * (one(T) - T(k - 1) / T(p)) for k in 1:p]
    γ = [τT * T(k) / T(p) for k in 1:p]
    Schedule{T}(β, γ)
end
