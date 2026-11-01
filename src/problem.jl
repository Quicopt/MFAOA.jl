# problem.jl — the two ways to state a problem, and the conversion between them.

const _SENSES = (:min, :max)

_check_sense(sense::Symbol) =
    sense in _SENSES || throw(ArgumentError("sense must be :min or :max, got :$sense"))

"""
    IsingProblem(h, J; Δ = nothing, offset = 0, sense = :min)

A binary quadratic program in Ising form: minimize (or, with `sense = :max`,
maximize)

    h' * σ + Σ_{i<j} J[i,j] σ[i] σ[j] + offset      over σ[i] ∈ {-1, +1}.

`J` must be symmetric with a zero diagonal (`σ[i]^2 = 1` would turn a diagonal
entry into a constant; put it into `offset`). `Δ` are the per-variable driver
strengths of the algorithm, all positive; `nothing` means `Δ[i] = 1` for all
`i`, which is the usual choice and the faster code path.

The fields `h`, `J`, `offset` always hold the problem in minimization form.
For `sense = :max` they are the negated inputs, and [`objective`](@ref)
flips the sign back, so objective values are always reported in the sense you
asked for.
"""
struct IsingProblem{T<:AbstractFloat,D<:Union{Nothing,Vector{T}}}
    h::Vector{T}
    J::Matrix{T}
    Δ::D
    offset::T
    sense::Symbol
end

function IsingProblem(h::AbstractVector, J::AbstractMatrix;
                      Δ::Union{Nothing,AbstractVector} = nothing,
                      offset::Real = 0, sense::Symbol = :min)
    _check_sense(sense)
    T = float(promote_type(eltype(h), eltype(J), typeof(offset),
                           Δ === nothing ? Bool : eltype(Δ)))
    n = length(h)
    n ≥ 1 || throw(ArgumentError("the problem needs at least one variable"))
    size(J) == (n, n) ||
        throw(ArgumentError("J must be $(n)×$(n) to match h, got $(size(J))"))
    @inbounds for j in 1:n
        iszero(J[j, j]) ||
            throw(ArgumentError("J must have a zero diagonal; J[$j,$j] = $(J[j, j]) is a constant (σ² = 1), add it to offset"))
        for i in 1:(j - 1)
            J[i, j] == J[j, i] ||
                throw(ArgumentError("J must be symmetric; J[$i,$j] = $(J[i, j]) but J[$j,$i] = $(J[j, i])"))
        end
    end
    if Δ !== nothing
        length(Δ) == n ||
            throw(ArgumentError("Δ must have one entry per variable, got $(length(Δ)) for n = $n"))
        all(>(0), Δ) || throw(ArgumentError("all driver strengths Δ[i] must be positive"))
    end
    sign = sense === :max ? -one(T) : one(T)
    hT = Vector{T}(undef, n)
    JT = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n
        hT[i] = sign * h[i]
    end
    @inbounds for j in 1:n, i in 1:n
        JT[i, j] = sign * J[i, j]
    end
    ΔT = Δ === nothing ? nothing : Vector{T}(Δ)
    IsingProblem{T,typeof(ΔT)}(hT, JT, ΔT, sign * T(offset), sense)
end

nvariables(prob::IsingProblem) = length(prob.h)
Base.eltype(::IsingProblem{T}) where {T} = T

"""
    objective(prob, σ)

The problem's objective value at the spin configuration `σ` (entries ±1),
in the sense the problem was stated in and including its constant offset.
"""
function objective(prob::IsingProblem{T}, σ::AbstractVector) where {T}
    E = energy(prob.h, prob.J, σ) + prob.offset
    prob.sense === :max ? -E : E
end

"""
    isflipsymmetric(prob) -> Bool

`true` when the objective is invariant under flipping all spins at once,
`σ → -σ`, which is the case when all local fields `h` vanish (up to rounding
relative to the size of `J`). The algorithm cannot solve such a problem: with
no field to break the symmetry the spins never leave their starting point.
Fix one variable to break the symmetry before solving.
"""
function isflipsymmetric(prob::IsingProblem{T}) where {T}
    hscale = maximum(abs, prob.h)
    jscale = maximum(abs, prob.J)
    hscale ≤ sqrt(eps(T)) * jscale
end

function Base.show(io::IO, prob::IsingProblem{T}) where {T}
    print(io, "IsingProblem{$T}: $(nvariables(prob)) variables, sense = :$(prob.sense)")
end

"""
    QUBOProblem(Q; c = zeros(n), constant = 0, sense = :min)

A binary quadratic program in QUBO form: minimize (or, with `sense = :max`,
maximize)

    x' * Q * x + c' * x + constant      over x[i] ∈ {0, 1}.

`Q` is used exactly as given: a symmetric matrix counts each off-diagonal pair
twice, an upper-triangular matrix once. Diagonal entries are allowed; since
`x[i]^2 = x[i]` they act as linear terms. [`solve`](@ref) converts the problem
to Ising form with `x = (1 + σ) / 2` (see [`IsingProblem(::QUBOProblem)`](@ref))
and reports the solution both as `x` and as `σ`.
"""
struct QUBOProblem{T<:AbstractFloat}
    Q::Matrix{T}
    c::Vector{T}
    constant::T
    sense::Symbol
end

function QUBOProblem(Q::AbstractMatrix; c::AbstractVector = zeros(Bool, size(Q, 1)),
                     constant::Real = 0, sense::Symbol = :min)
    _check_sense(sense)
    n = size(Q, 1)
    n ≥ 1 || throw(ArgumentError("the problem needs at least one variable"))
    size(Q) == (n, n) || throw(ArgumentError("Q must be square, got $(size(Q))"))
    length(c) == n || throw(ArgumentError("c must have one entry per variable, got $(length(c)) for n = $n"))
    T = float(promote_type(eltype(Q), eltype(c), typeof(constant)))
    QUBOProblem{T}(Matrix{T}(Q), Vector{T}(c), T(constant), sense)
end

nvariables(prob::QUBOProblem) = length(prob.c)
Base.eltype(::QUBOProblem{T}) where {T} = T

"""
    objective(prob::QUBOProblem, x)

The objective value `x' * Q * x + c' * x + constant` at `x` (entries 0 or 1).
Maximization problems report the value as stated, without sign change.
"""
function objective(prob::QUBOProblem{T}, x::AbstractVector) where {T}
    n = nvariables(prob)
    E = prob.constant
    @inbounds for j in 1:n
        xj = T(x[j])
        iszero(xj) && continue
        acc = prob.c[j]
        @simd for i in 1:n
            acc = muladd(prob.Q[i, j], T(x[i]), acc)
        end
        E += xj * acc
    end
    E
end

function Base.show(io::IO, prob::QUBOProblem{T}) where {T}
    print(io, "QUBOProblem{$T}: $(nvariables(prob)) variables, sense = :$(prob.sense)")
end

"""
    IsingProblem(prob::QUBOProblem; Δ = nothing)

The same problem in Ising form. With `x = (1 + σ) / 2`, a term `Q[i,j] x[i] x[j]`
with `i ≠ j` contributes `Q[i,j] / 4` to the coupling of the pair, to both
fields, and to the offset; the linear total `c[i] + Q[i,i]` contributes half of
itself to `h[i]` and half to the offset.
"""
function IsingProblem(prob::QUBOProblem{T}; Δ::Union{Nothing,AbstractVector} = nothing) where {T}
    n = nvariables(prob)
    Q = prob.Q
    h = Vector{T}(undef, n)
    J = zeros(T, n, n)
    offset = prob.constant
    @inbounds for i in 1:n
        ℓ = prob.c[i] + Q[i, i]
        h[i] = ℓ / 2
        offset += ℓ / 2
    end
    @inbounds for j in 1:n, i in 1:(j - 1)
        q = (Q[i, j] + Q[j, i]) / 4
        J[i, j] = q
        J[j, i] = q
        h[i] += q
        h[j] += q
        offset += q
    end
    IsingProblem(h, J; Δ = Δ, offset = offset, sense = prob.sense)
end
