# schedule.jl — the two angle sequences that drive the algorithm.

"""
    Schedule(β, γ)

The two parameter sequences of the algorithm, one value per layer. `β[k]` sets
the driver rotation of layer `k`: spin `i` turns about x̂ by `2β[k]Δ[i]`.
`γ[k]` sets the problem rotation: spin `i` turns about ẑ by `2γ[k]m[i]/s`,
where `m[i]` is the field the spin feels and `s` is the problem's
[`fieldscale`](@ref). The two vectors must have the same length, which is the
number of layers `p`. [`linear_schedule`](@ref) builds the standard choice;
any other pair of vectors can be passed directly.
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

The standard linear schedule with `p` layers and time step `τ`: the problem
weight grows as `γ[k] = τ k / p` while the driver weight shrinks as
`β[k] = τ (1 - (k - 1) / p)`, for `k = 1:p` (Eq. 13 of the paper). The time
step sets how far each layer rotates the spins. Because `γ` is measured in
units of the problem's [`fieldscale`](@ref), the same `τ` suits problems of
any size and units; `τ` between 0.25 and 0.5 works well for typical
instances. Larger `p` makes the sweep slower and the result more stable.
"""
function linear_schedule(p::Integer, τ::Real = 0.5; T::Type{<:AbstractFloat} = Float64)
    p ≥ 1 || throw(ArgumentError("need at least one layer, got p = $p"))
    τT = T(τ)
    β = [τT * (one(T) - T(k - 1) / T(p)) for k in 1:p]
    γ = [τT * T(k) / T(p) for k in 1:p]
    Schedule{T}(β, γ)
end
