# kernel.jl — the mean-field AOA layer loop. No dependencies, no allocation.
#
# The problem is to minimize
#
#     E(σ) = Σᵢ hᵢ σᵢ + Σ_{i<j} Jᵢⱼ σᵢ σⱼ,     σᵢ = ±1,
#
# with J symmetric and zero on the diagonal. Each binary variable becomes a
# classical unit spin nᵢ = (nˣ, nʸ, nᶻ), started at (1, 0, 0). Layer k rotates
# every spin twice:
#
#   1. about ẑ by the angle 2γₖ mᵢ, where mᵢ = hᵢ + Σⱼ Jᵢⱼ nⱼᶻ is the effective
#      field spin i feels from the problem (one J contraction per layer);
#   2. about x̂ by the angle 2βₖ Δᵢ, where Δᵢ > 0 is the driver strength.
#
# These are Eqs. 10–12 of Misra-Spieldenner et al., PRX Quantum 4, 030335
# (2023), with (h, J) → (−h, −J): the paper minimizes −Σᵢ[hᵢ + Σ_{j>i} Jᵢⱼσⱼ]σᵢ,
# we minimize +Σᵢ[...]σᵢ, so the ẑ rotation runs the other way round. The
# rotation sense follows from the precession ∂ₜn = n × B with B = −2∇ₙH: a
# positive effective field mᵢ pushes nᵢᶻ negative, which lowers hᵢσᵢ.
#
# After the last layer, σᵢ = sign(nᵢᶻ). A spin that ends exactly on the
# equator takes the sign that lowers the energy in its final field,
# σᵢ = −sign(mᵢ), and +1 if that field vanishes too.
#
# Spins are stored as three separate vectors (x, y, z) so the J contraction
# streams over a contiguous z and the rotations stream over x, y, z.

"""
    SpinState{T}(n)

The `n` classical unit spins of the mean-field dynamics, stored as three
vectors `x`, `y`, `z` of components. A fresh state has every spin at
`(1, 0, 0)`, the starting point of the algorithm.
"""
struct SpinState{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
end

SpinState{T}(n::Integer) where {T<:AbstractFloat} =
    SpinState{T}(ones(T, n), zeros(T, n), zeros(T, n))

Base.length(s::SpinState) = length(s.x)

"""
    reset!(s::SpinState)

Put every spin back to `(1, 0, 0)`.
"""
function reset!(s::SpinState{T}) where {T}
    fill!(s.x, one(T))
    fill!(s.y, zero(T))
    fill!(s.z, zero(T))
    s
end

"""
    magnetization!(m, h, J, z)

Compute the effective field `m = h + J * z` without allocating. This is the
one O(n²) operation of each layer.
"""
function magnetization!(m::AbstractVector{T}, h::AbstractVector, J::AbstractMatrix,
                        z::AbstractVector) where {T}
    n = length(m)
    @inbounds for i in 1:n
        m[i] = h[i]
    end
    @inbounds for j in 1:n
        zj = z[j]
        @simd for i in 1:n
            m[i] = muladd(J[i, j], zj, m[i])
        end
    end
    m
end

# One layer of rotations given the effective field m. Per-spin driver strengths.
function rotate!(s::SpinState, m::AbstractVector, γ::Real, β::Real, Δ::AbstractVector)
    @inbounds for i in eachindex(m)
        sp, cp = sincos(2 * γ * m[i])
        x = s.x[i]
        y = s.y[i]
        x, y = cp * x - sp * y, sp * x + cp * y
        sd, cd = sincos(2 * β * Δ[i])
        z = s.z[i]
        s.x[i] = x
        s.y[i] = cd * y + sd * z
        s.z[i] = cd * z - sd * y
    end
    s
end

# Same with the uniform driver Δᵢ = 1: the driver angle is shared by all spins.
function rotate!(s::SpinState, m::AbstractVector, γ::Real, β::Real, ::Nothing)
    sd, cd = sincos(2 * β)
    @inbounds for i in eachindex(m)
        sp, cp = sincos(2 * γ * m[i])
        x = s.x[i]
        y = s.y[i]
        x, y = cp * x - sp * y, sp * x + cp * y
        z = s.z[i]
        s.x[i] = x
        s.y[i] = cd * y + sd * z
        s.z[i] = cd * z - sd * y
    end
    s
end

"""
    step!(s::SpinState, m, h, J, Δ, β_k, γ_k)

Apply one layer: compute the effective field into the buffer `m`, rotate every
spin about ẑ by `2γ_k * m[i]`, then about x̂ by `2β_k * Δ[i]`. Pass `Δ = nothing`
for the uniform driver `Δ[i] = 1`. Allocates nothing.
"""
function step!(s::SpinState, m::AbstractVector, h::AbstractVector, J::AbstractMatrix,
               Δ, β_k::Real, γ_k::Real)
    magnetization!(m, h, J, s.z)
    rotate!(s, m, γ_k, β_k, Δ)
end

"""
    evolve!(s::SpinState, m, h, J, Δ, β, γ)

Run all layers `k = 1:length(β)` of [`step!`](@ref) on the state `s`, in place.
`β` and `γ` must have the same length. Allocates nothing.
"""
function evolve!(s::SpinState, m::AbstractVector, h::AbstractVector, J::AbstractMatrix,
                 Δ, β::AbstractVector, γ::AbstractVector)
    @inbounds for k in eachindex(β, γ)
        step!(s, m, h, J, Δ, β[k], γ[k])
    end
    s
end

"""
    readout!(σ, s::SpinState, m)

Round the spins to `σ[i] = sign(s.z[i])`. A spin with `z == 0` takes
`-sign(m[i])`, the choice that lowers the energy in its field `m`, and `+1` if
`m[i] == 0` as well. `m` must hold the effective field of the state `s`.
"""
function readout!(σ::AbstractVector{<:Integer}, s::SpinState, m::AbstractVector)
    @inbounds for i in eachindex(σ)
        z = s.z[i]
        σ[i] = z > 0 ? 1 : z < 0 ? -1 : (m[i] > 0 ? -1 : 1)
    end
    σ
end

"""
    energy(h, J, v)

The value `h' * v + Σ_{i<j} J[i,j] v[i] v[j]` for any real vector `v`, for
example the rounded spins `σ` or the continuous `z` components. Allocates
nothing.
"""
function energy(h::AbstractVector{T}, J::AbstractMatrix, v::AbstractVector) where {T}
    n = length(h)
    E = zero(T)
    @inbounds for j in 1:n
        acc = zero(T)
        @simd for i in 1:(j - 1)
            acc = muladd(J[i, j], T(v[i]), acc)
        end
        E += T(v[j]) * (h[j] + acc)
    end
    E
end
