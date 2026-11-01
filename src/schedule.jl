"""
    Schedule(β, γ)

The two parameter sequences of the algorithm, one value per layer: `β[k]` sets
the driver rotation of layer `k` (its angle is `2β[k]Δ[i]`), `γ[k]` the problem
rotation (its angle is `2γ[k]m[i]`). The two vectors must have the same length,
which is the number of layers `p`. [`linear_schedule`](@ref) builds the
standard choice; any other pair of vectors can be passed directly.
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

function Schedule(β::AbstractVector, γ::AbstractVector)
    T = float(promote_type(eltype(β), eltype(γ)))
    Schedule{T}(Vector{T}(β), Vector{T}(γ))
end

Base.length(s::Schedule) = length(s.β)

"""
    linear_schedule(p, τ = 0.5; T = Float64)

The standard linear schedule with `p` layers and time step `τ`: the problem
weight grows as `γ[k] = τ k / p` while the driver weight shrinks as
`β[k] = τ (1 - (k - 1) / p)`, for `k = 1:p` (Eq. 13 of the paper). The time
step sets how far each layer rotates the spins; `τ ≈ 0.5` keeps the dynamics
regular for typical instances, and larger `p` makes the sweep slower and the
result more stable.
"""
function linear_schedule(p::Integer, τ::Real = 0.5; T::Type{<:AbstractFloat} = Float64)
    p ≥ 1 || throw(ArgumentError("need at least one layer, got p = $p"))
    τT = T(τ)
    β = [τT * (one(T) - T(k - 1) / T(p)) for k in 1:p]
    γ = [τT * T(k) / T(p) for k in 1:p]
    Schedule{T}(β, γ)
end
