# kernel.jl — the mean-field AOA layer loop. Allocates nothing.
#
# The problem is to minimize
#
#     E(σ) = Σᵢ hᵢ σᵢ + Σ_{i<j} Jᵢⱼ σᵢ σⱼ,     σᵢ = ±1,
#
# with J symmetric and zero on the diagonal. Each binary variable becomes a
# classical unit spin nᵢ = (nˣ, nʸ, nᶻ), started at (1, 0, 0). Layer k rotates
# every spin twice:
#
#   1. about ẑ by the angle 2γₖ mᵢ / s, where mᵢ = hᵢ + Σⱼ Jᵢⱼ nⱼᶻ is the
#      effective field spin i feels from the problem (one product of J with a
#      vector per layer) and s is the problem's field scale;
#   2. about x̂ by the angle 2βₖ Δᵢ, where Δᵢ > 0 is the driver strength.
#
# These are Eqs. 10–12 of Misra-Spieldenner et al., PRX Quantum 4, 030335
# (2023), with (h, J) → (−h, −J): the paper minimizes −Σᵢ[hᵢ + Σ_{j>i} Jᵢⱼσⱼ]σᵢ,
# we minimize +Σᵢ[...]σᵢ, so the ẑ rotation runs the other way round. The
# rotation sense follows from the precession ∂ₜn = n × B with B = −2∇ₙH: a
# positive effective field mᵢ pushes nᵢᶻ negative, which lowers hᵢσᵢ.
#
# Two things enter the ẑ angle only as a common factor, so they cost one
# scalar per layer instead of a rescaled or negated copy of the problem: the
# field scale s, which makes the dynamics independent of the units of the
# objective, and the sense. Maximizing E is minimizing −E, whose field is −mᵢ,
# so a maximization rotates by −2γₖ mᵢ / s.
#
# After the last layer, σᵢ = sign(nᵢᶻ). A spin that ends exactly on the
# equator takes the sign that improves the objective in its final field, and
# +1 if that field vanishes too.
#
# Spins are stored as three separate vectors (x, y, z) so the product with J
# reads a contiguous z and the rotations stream over x, y, z.

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

"""
    length(s::SpinState)

The number of spins.
"""
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
    Workspace(prob)

All memory the algorithm needs for `prob`: the spin state `state`, the field
buffer `m` and the rounded spins `σ`. Allocate it once and reuse it across
solves of problems with the same number of variables.
"""
struct Workspace{T<:AbstractFloat}
    state::SpinState{T}
    m::Vector{T}
    σ::Vector{Int8}
end

function Workspace(prob::IsingProblem{T}) where {T}
    n = nvariables(prob)
    Workspace{T}(SpinState{T}(n), Vector{T}(undef, n), Vector{Int8}(undef, n))
end

"""
    magnetization!(m, h, J, z) -> m

The effective field `m = h + J * z`, computed in place by `mul!`, so a dense
`J` goes to BLAS and a sparse one to the sparse product. This is the one
operation of each layer whose cost grows faster than `n`.
"""
magnetization!(m::AbstractVector, h::AbstractVector, J::AbstractMatrix, z::AbstractVector) =
    mul!(copyto!(m, h), J, z, true, true)

"""
    rotate!(s::SpinState, m, γ, β, Δ) -> s

Rotate every spin `i` about ẑ by `2γ * m[i]`, then about x̂ by `2β * Δ[i]`;
`Δ = nothing` stands for `Δ[i] = 1`, which shares one driver angle among all
spins. Here `γ` already contains the factor that converts the field into the
problem's units and sense (see [`step!`](@ref)).
"""
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
    _field_unit(prob) -> ±1/s

The factor that turns a field of `prob`, in the units and sense the problem
was stated in, into the field the dynamics minimize, in units of
[`fieldscale`](@ref)`(prob)`.
"""
_field_unit(prob::IsingProblem{T}) where {T} = _sense_sign(prob.sense, T) / prob.scale

"""
    step!(ws::Workspace, prob::IsingProblem, β_k, γ_k) -> ws

Apply one layer to the spins in `ws.state`: compute their effective field
into `ws.m`, rotate every spin `i` about ẑ by `2γ_k * m[i] / s` (with `s` the
problem's [`fieldscale`](@ref), and the opposite sign for a maximization),
then about x̂ by `2β_k * Δ[i]`. Allocates nothing.
"""
function step!(ws::Workspace, prob::IsingProblem, β_k::Real, γ_k::Real)
    magnetization!(ws.m, prob.h, prob.J, ws.state.z)
    rotate!(ws.state, ws.m, γ_k * _field_unit(prob), β_k, prob.Δ)
    ws
end

"""
    evolve!(ws::Workspace, prob::IsingProblem, sched::Schedule) -> ws

Run every layer of `sched` with [`step!`](@ref), starting from the spins in
`ws.state` as they are. Allocates nothing.
"""
function evolve!(ws::Workspace, prob::IsingProblem, sched::Schedule)
    @inbounds for k in eachindex(sched.β)
        step!(ws, prob, sched.β[k], sched.γ[k])
    end
    ws
end

"""
    readout!(σ, s::SpinState, m, prob) -> σ

Round the spins to `σ[i] = sign(s.z[i])`. A spin with `z == 0` takes the sign
that improves the objective of `prob` in its field `m[i]`, and `+1` if
`m[i] == 0` as well. `m` must hold the effective field of the state `s`.
"""
function readout!(σ::AbstractVector{<:Integer}, s::SpinState, m::AbstractVector, prob::IsingProblem)
    sign = _sense_sign(prob.sense, eltype(m))
    @inbounds for i in eachindex(σ)
        z = s.z[i]
        σ[i] = z > 0 ? 1 : z < 0 ? -1 : (sign * m[i] > 0 ? -1 : 1)
    end
    σ
end

"""
    energy(h, J, v)

The value `h' * v + Σ_{i<j} J[i,j] v[i] v[j] = h' * v + v' * J * v / 2` for any
real vector `v`, for example the rounded spins `σ` or the continuous `z`
components. Allocates nothing.
"""
energy(h::AbstractVector, J::AbstractMatrix, v::AbstractVector) = dot(h, v) + dot(v, J, v) / 2
