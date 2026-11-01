# problem.jl — the two ways to state a problem, and the conversion between them.
#
# Both problem types keep the vectors and matrices they are given, without
# copying them, when the element type already matches: a problem built from a
# large matrix costs no memory beyond that matrix. The objective is stored in
# the sense it was stated in. The kernel turns a maximization into a
# minimization by reversing the problem rotation (see kernel.jl), so no negated
# copy is needed either.

const _SENSES = (:min, :max)

"""
    _check_sense(sense::Symbol) -> true

Throw an `ArgumentError` unless `sense` is `:min` or `:max`.
"""
_check_sense(sense::Symbol) =
    sense in _SENSES || throw(ArgumentError("sense must be :min or :max, got :$sense"))

"""
    _sense_sign(sense::Symbol, T) -> T

`+1` for `:min` and `-1` for `:max`: the factor that turns the stated
objective into the one to be minimized.
"""
_sense_sign(sense::Symbol, ::Type{T}) where {T} = sense === :max ? -one(T) : one(T)

"""
    _entries(A::AbstractMatrix)

The entries of `A` that can be nonzero: all of them for a dense matrix, the
stored values for a sparse one. Reductions over `_entries(A)` skip the
structural zeros of a sparse matrix.
"""
_entries(A::AbstractMatrix) = A
_entries(A::SparseMatrixCSC) = nonzeros(A)

"""
    _rms_field(h, J) -> s

The root mean square of the field `mᵢ = hᵢ + Σⱼ Jᵢⱼσⱼ` over all `i` and over
uniformly random spins `σ`. The cross terms average out, which leaves
`s = sqrt((Σᵢ hᵢ² + Σᵢⱼ Jᵢⱼ²) / n)`.
"""
_rms_field(h::AbstractVector, J::AbstractMatrix) =
    sqrt((sum(abs2, h) + sum(abs2, _entries(J))) / length(h))

"""
    _check_couplings(J, n)

Throw an `ArgumentError` unless `J` is `n × n`, zero on the diagonal and
symmetric.
"""
function _check_couplings(J::AbstractMatrix, n::Integer)
    size(J) == (n, n) ||
        throw(ArgumentError("J must be $(n)×$(n) to match h, got $(size(J))"))
    for i in 1:n
        iszero(J[i, i]) ||
            throw(ArgumentError("J must be zero on the diagonal, but J[$i,$i] = $(J[i, i]). Since σ[i]^2 = 1, a diagonal entry only adds a constant: add it to offset instead."))
    end
    if !issymmetric(J)
        i, j = _asymmetric_pair(J)
        # At most ten interpolated pieces, so that `string` resolves statically
        # and the constructor compiles with JuliaC's --trim.
        throw(ArgumentError("J must be symmetric, but J[i,j] = $(J[i, j]) and J[j,i] = $(J[j, i]) at (i, j) = $((i, j))"))
    end
    nothing
end

"""
    _asymmetric_pair(J) -> (i, j)

An index pair with `J[i,j] != J[j,i]`, for the error message about a matrix
that is not symmetric; `(0, 0)` if there is none. For a sparse matrix only the
stored entries are visited: every asymmetric pair has at least one of them.
"""
function _asymmetric_pair(J::AbstractMatrix)
    for j in axes(J, 2), i in axes(J, 1)
        J[i, j] == J[j, i] || return (i, j)
    end
    (0, 0)
end

function _asymmetric_pair(J::SparseMatrixCSC)
    rows, cols, _ = findnz(J)
    for (i, j) in zip(rows, cols)
        J[i, j] == J[j, i] || return (i, j)
    end
    (0, 0)
end

"""
    _check_drivers(Δ, n)

Throw an `ArgumentError` unless `Δ` is `nothing` or holds `n` positive driver
strengths.
"""
function _check_drivers(Δ::Union{Nothing,AbstractVector}, n::Integer)
    Δ === nothing && return nothing
    length(Δ) == n ||
        throw(ArgumentError("Δ must have one entry per variable, got $(length(Δ)) for n = $n"))
    all(>(0), Δ) || throw(ArgumentError("all driver strengths Δ[i] must be positive"))
    nothing
end

"""
    IsingProblem(h, J; Δ = nothing, offset = 0, sense = :min, scale = nothing)

A binary quadratic program in Ising form, in variables `σ[i]` that are -1 or
+1: minimize (or, with `sense = :max`, maximize)

    h' * σ + Σ_{i<j} J[i,j] σ[i] σ[j] + offset.

`J` must be symmetric with a zero diagonal. Since `σ[i]^2 = 1`, a diagonal
entry would only add a constant; put it into `offset` instead. `J` can be a
dense `Matrix`, a `SparseMatrixCSC`, which is much faster when most entries
are zero, or any other matrix that `LinearAlgebra.mul!` can multiply with a
vector, such as a `Symmetric` view. `h` and `J` are used as they are, without
a copy, when their element type is already the problem's. Changing them
afterwards changes the problem, except for its [`fieldscale`](@ref), which is
computed when the problem is built.

Two keywords tune the algorithm and rarely need to be set:

- `Δ`: one positive factor per variable on the angles `β` of the
  [`Schedule`](@ref). The default, `nothing`, means 1 for every variable and
  takes a faster code path.
- `scale`: the unit in which the schedule measures the slope of the
  objective; see [`fieldscale`](@ref). The default makes the solution
  independent of the units of `h` and `J`; `scale = 1` uses those units
  themselves.
"""
struct IsingProblem{T<:AbstractFloat,M<:AbstractMatrix{T},D<:Union{Nothing,Vector{T}}}
    h::Vector{T}
    J::M
    Δ::D
    offset::T
    sense::Symbol
    scale::T
end

function IsingProblem(h::AbstractVector, J::AbstractMatrix;
                      Δ::Union{Nothing,AbstractVector} = nothing,
                      offset::Real = 0, sense::Symbol = :min,
                      scale::Union{Nothing,Real} = nothing)
    _check_sense(sense)
    T = float(promote_type(eltype(h), eltype(J), typeof(offset),
                           Δ === nothing ? Bool : eltype(Δ)))
    n = length(h)
    n ≥ 1 || throw(ArgumentError("the problem needs at least one variable"))
    _check_couplings(J, n)
    _check_drivers(Δ, n)
    hT = convert(Vector{T}, h)
    JT = convert(AbstractMatrix{T}, J)
    ΔT = Δ === nothing ? nothing : convert(Vector{T}, Δ)
    s = scale === nothing ? _rms_field(hT, JT) : T(scale)
    scale === nothing && iszero(s) && (s = one(T))     # an all-zero objective
    s > 0 || throw(ArgumentError("scale must be positive, got $scale"))
    IsingProblem{T,typeof(JT),typeof(ΔT)}(hT, JT, ΔT, T(offset), sense, s)
end

"""
    nvariables(prob)

The number of binary variables of `prob`.
"""
nvariables(prob::IsingProblem) = length(prob.h)

"""
    eltype(prob)

The floating-point type the problem is stored and solved in.
"""
Base.eltype(::IsingProblem{T}) where {T} = T

"""
    fieldscale(prob::IsingProblem) -> s

The unit in which the schedule measures the slope of the objective of `prob`.

The algorithm moves each variable according to the slope of the objective in
that variable, `m[i] = h[i] + Σⱼ J[i,j] z[j]`, where `z[j]`, between -1 and +1,
is the current value of variable `j`. In step `k`, the angle that this slope
sets is `2γ[k] m[i] / s` (see [`Schedule`](@ref)). Unless `scale` was given
when the problem was built, `s` is the root mean square of `m[i]` over all
variables `i` and over random values ±1 of `z`:

    s = sqrt((Σᵢ hᵢ² + Σᵢⱼ Jᵢⱼ²) / n).

Multiplying `h` and `J` by a constant multiplies `s` by the same constant, so
the algorithm, and the solution, do not depend on the units of the objective.
"""
fieldscale(prob::IsingProblem) = prob.scale

"""
    objective(prob, σ)

The objective value of `prob` at `σ`, a vector with entries -1 and +1,
including the offset.
"""
objective(prob::IsingProblem, σ::AbstractVector) = energy(prob.h, prob.J, σ) + prob.offset

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
`x[i]^2 = x[i]` they act as linear terms. `Q` can be dense or sparse, and is
kept without copying when its element type is already the problem's.
[`solve`](@ref) converts the problem to Ising form with `x = (1 + σ) / 2`
(see [`IsingProblem(::QUBOProblem)`](@ref)) and reports the solution both as
`x` and as `σ`.
"""
struct QUBOProblem{T<:AbstractFloat,M<:AbstractMatrix{T}}
    Q::M
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
    QT = convert(AbstractMatrix{T}, Q)
    QUBOProblem{T,typeof(QT)}(QT, convert(Vector{T}, c), T(constant), sense)
end

nvariables(prob::QUBOProblem) = length(prob.c)
Base.eltype(::QUBOProblem{T}) where {T} = T

"""
    objective(prob::QUBOProblem, x)

The objective value `x' * Q * x + c' * x + constant` at `x` (entries 0 or 1).
Maximization problems report the value as stated, without sign change.
"""
objective(prob::QUBOProblem, x::AbstractVector) =
    dot(x, prob.Q, x) + dot(prob.c, x) + prob.constant

function Base.show(io::IO, prob::QUBOProblem{T}) where {T}
    print(io, "QUBOProblem{$T}: $(nvariables(prob)) variables, sense = :$(prob.sense)")
end

"""
    IsingProblem(prob::QUBOProblem; Δ = nothing, scale = nothing)

The same problem in Ising form, with `x = (1 + σ) / 2`. A pair term
`Q[i,j] x[i] x[j] + Q[j,i] x[j] x[i]` becomes the coupling
`J[i,j] = (Q[i,j] + Q[j,i]) / 4` and contributes the same amount to both
fields and to the offset. The weight of `x[i]` alone, `c[i] + Q[i,i]`, goes
half into `h[i]` and half into the offset. `J` is sparse if `Q` is.
"""
function IsingProblem(prob::QUBOProblem{T}; Δ::Union{Nothing,AbstractVector} = nothing,
                      scale::Union{Nothing,Real} = nothing) where {T}
    n = nvariables(prob)
    J = _ising_couplings(prob.Q)
    linear = [prob.c[i] + prob.Q[i, i] for i in 1:n]   # the weight of x[i] alone
    coupling_sums = J * ones(T, n)                      # Σⱼ J[i,j]
    h = linear ./ 2 .+ coupling_sums
    offset = prob.constant + sum(linear) / 2 + sum(coupling_sums) / 2
    IsingProblem(h, J; Δ, offset, sense = prob.sense, scale)
end

"""
    _ising_couplings(Q) -> J

The Ising couplings `J[i,j] = (Q[i,j] + Q[j,i]) / 4` for `i ≠ j`, with a zero
diagonal: a new dense matrix for a dense `Q`, a new sparse one for a sparse
`Q`.
"""
function _ising_couplings(Q::AbstractMatrix{T}) where {T}
    n = size(Q, 1)
    J = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        J[i, j] = i == j ? zero(T) : (Q[i, j] + Q[j, i]) / 4
    end
    J
end

function _ising_couplings(Q::SparseMatrixCSC{T}) where {T}
    rows, cols, vals = findnz(Q)
    offdiagonal = rows .!= cols
    i = rows[offdiagonal]
    j = cols[offdiagonal]
    v = vals[offdiagonal] ./ 4
    # Both orientations of every stored entry; `sparse` adds up the duplicates,
    # which gives (Q[i,j] + Q[j,i]) / 4 at (i, j) and at (j, i).
    dropzeros!(sparse(vcat(i, j), vcat(j, i), vcat(v, v), size(Q)...))
end
