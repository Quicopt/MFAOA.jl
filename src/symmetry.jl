# symmetry.jl — finding the groups of variables that the algorithm cannot move.
#
# Two variables are linked when a nonzero coupling J[i,j] joins them. A group
# is a set of variables linked to each other, directly or through others, and
# to no variable outside it. If h vanishes on a group of two or more variables,
# flipping the whole group (σ → -σ on the group) leaves the objective
# unchanged. The algorithm cannot handle such a group: every spin starts with
# z = 0, so the field h + J z is zero on the group at the start, the rotations
# leave the group's spins where they are, and the field stays zero for good.
# A group of a single variable with h = 0 is harmless: the objective does not
# depend on that variable, and the readout sets it to +1.
#
# solve! runs the check before every solve, in the workspace's memory, so it
# sees h and J as they are at that moment, even if they were changed in place
# after the problem was built. It costs one pass over the couplings, as much
# as one to three steps of the algorithm (measured on an Apple M4, n = 1000
# to 5000). The groups are found by union-find in `ws.group`, with `ws.σ` as
# scratch for one flag byte per group.

"""
    _root!(parent, i) -> r

The representative of the group of variable `i` in the union-find array
`parent`, halving the path to it on the way.
"""
function _root!(parent::AbstractVector{Int}, i::Int)
    @inbounds while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    i
end

"""
    _link!(parent, i, r) -> r′

Merge the group of variable `i` into the group with representative `r`, and
return the representative of the merged group. The smaller of the two
representatives becomes it, so `parent[i] ≤ i` holds throughout.
"""
function _link!(parent::AbstractVector{Int}, i::Int, r::Int)
    @inbounds parent[i] == r && return r     # the common case: already merged
    a = _root!(parent, i)
    @inbounds if a < r
        parent[r] = a
        return a
    elseif r < a
        parent[a] = r
    end
    r
end

"""
    _link_couplings!(parent, J) -> Jmax

Link every pair of variables joined by a nonzero coupling, and return the
largest `|J[i,j]|`. Column `j` of `J` lists the couplings of variable `j`;
only the upper triangle of a dense `J` is read, since `J` is symmetric, and
only the stored entries of a sparse one.
"""
function _link_couplings!(parent::AbstractVector{Int}, J::AbstractMatrix{T}) where {T}
    Jmax = zero(T)
    @inbounds for j in axes(J, 2)
        r = _root!(parent, j)
        for i in 1:(j - 1)
            v = J[i, j]
            iszero(v) && continue
            Jmax = max(Jmax, abs(v))
            r = _link!(parent, i, r)
        end
    end
    Jmax
end

function _link_couplings!(parent::AbstractVector{Int}, J::SparseMatrixCSC{T}) where {T}
    Jmax = zero(T)
    rows = rowvals(J)
    vals = nonzeros(J)
    @inbounds for j in axes(J, 2)
        r = _root!(parent, j)
        for k in nzrange(J, j)
            v = vals[k]
            iszero(v) && continue
            Jmax = max(Jmax, abs(v))
            r = _link!(parent, rows[k], r)
        end
    end
    Jmax
end

"""
    _mark_groups!(parent, flags, h, J) -> count

Sort the variables into groups and return the number of groups the algorithm
cannot move: groups of two or more variables on which `h` vanishes, up to
`sqrt(eps)` times the largest coupling. Afterwards `parent[i]` is the
representative of the group of variable `i`, which is its smallest variable,
and for a representative `r`, bit 1 of `flags[r]` says that the group has more
than one variable and bit 2 that `h` is nonzero on it. `parent` and `flags`
need one entry per variable. Allocates nothing.
"""
function _mark_groups!(parent::AbstractVector{Int}, flags::AbstractVector{Int8},
                       h::AbstractVector{T}, J::AbstractMatrix) where {T}
    n = length(h)
    @inbounds for i in 1:n
        parent[i] = i
        flags[i] = 0
    end
    tol = sqrt(eps(T)) * _link_couplings!(parent, J)
    count = 0
    @inbounds for i in 1:n
        # parent[i] ≤ i, and the entries below i already point to their
        # representatives, so one step reaches the representative of i.
        r = parent[parent[i]]
        parent[i] = r
        r == i || (flags[r] |= Int8(1))
        abs(h[i]) > tol && (flags[r] |= Int8(2))
    end
    @inbounds for r in 1:n
        count += (parent[r] == r) & (flags[r] == 1)
    end
    count
end

"""
    _flip_groups(prob::IsingProblem) -> Vector{Vector{Int}}

The groups of variables of `prob` that the algorithm cannot move (see
[`isflipsymmetric`](@ref)), each as its variable indices in increasing order,
the groups ordered by their first variable. Empty if there are none.
"""
function _flip_groups(prob::IsingProblem)
    n = nvariables(prob)
    parent = Vector{Int}(undef, n)
    flags = Vector{Int8}(undef, n)
    groups = Vector{Int}[]
    _mark_groups!(parent, flags, prob.h, prob.J) == 0 && return groups
    position = zeros(Int, n)        # where the group of representative r sits in `groups`
    for r in 1:n
        if parent[r] == r && flags[r] == 1
            push!(groups, Int[])
            position[r] = length(groups)
        end
    end
    for i in 1:n
        g = position[parent[i]]
        g > 0 && push!(groups[g], i)
    end
    groups
end

"""
    isflipsymmetric(prob) -> Bool

Whether `prob` has a group of variables that MFAOA cannot solve for: flipping
all variables of the group at once (`σ → -σ`, or `x → 1 - x` in a
[`QUBOProblem`](@ref)) leaves the objective unchanged. In such a group, no
term of the objective links a variable to one outside the group, and `h` is
zero on every variable of the group, up to rounding errors relative to the
largest coupling. When `h` is zero everywhere, as in Max-Cut, the whole
problem is such a group. A single variable that appears in no term of the
objective does not count: any value is optimal for it, and the solution sets
it to `σ = +1`.

The algorithm starts every variable at `σ = 0`, halfway between -1 and +1.
There the objective has zero slope in the variables of such a group, so they
would never move. [`solve`](@ref) throws an `ArgumentError` instead, which
names the groups. Fix one variable of each group to either value and solve for
the others. This loses nothing: flipping the group turns every solution into
one with the same objective value, so of each such pair, one solution remains.

To fix `σ[i] = s`, remove variable `i` from the problem, add `s * J[j,i]` to
`h[j]` for every other variable `j`, and add `s * h[i]` to the offset. In a
`QUBOProblem`, fixing `x[i] = v` adds `v * (Q[i,j] + Q[j,i])` to `c[j]` and
`v * (Q[i,i] + c[i])` to the constant.
"""
function isflipsymmetric(prob::IsingProblem)
    n = nvariables(prob)
    _mark_groups!(Vector{Int}(undef, n), Vector{Int8}(undef, n), prob.h, prob.J) > 0
end

isflipsymmetric(prob::QUBOProblem) = isflipsymmetric(IsingProblem(prob))

# The words in which `_flip_message` explains the problem, for each way of
# stating it. `fix` follows the advice to fix a variable.
const _ISING_WORDS = (flip = "σ → -σ", start = "0, halfway between -1 and +1",
                      values = "-1 or +1", fix = " (see `?isflipsymmetric` for how)")
const _QUBO_WORDS = (flip = "x → 1 - x", start = "1/2", values = "0 or 1",
                     fix = " (see `?isflipsymmetric` for how)")

"""
    _print_names(io, name, indices; most = 5)

Print the names `name(i)` of the variables `indices` as an English list,
`"a, b and c"`, showing at most `most` of them and counting the rest.
"""
function _print_names(io::IO, name, indices::AbstractVector{Int}; most::Int = 5)
    shown = min(length(indices), most)
    for k in 1:shown
        k > 1 && print(io, k == length(indices) ? " and " : ", ")
        print(io, name(indices[k]))
    end
    rest = length(indices) - shown
    rest > 0 && print(io, " and $rest more")
    nothing
end

"""
    _print_set(io, name, indices; most = 5)

Print the names `name(i)` of the variables `indices` as a set, `"{a, b, c}"`,
showing at most `most` of them and giving the size of a larger set:
`"{a, b, c, d, e, … (12 variables)}"`.
"""
function _print_set(io::IO, name, indices::AbstractVector{Int}; most::Int = 5)
    print(io, "{")
    for k in 1:min(length(indices), most)
        k > 1 && print(io, ", ")
        print(io, name(indices[k]))
    end
    length(indices) > most && print(io, ", … ($(length(indices)) variables)")
    print(io, "}")
    nothing
end

"""
    _flip_message(groups, n, name, words) -> String

Explain why a problem in `n` variables with the flip-symmetric `groups` (from
[`_flip_groups`](@ref)) cannot be solved, and what to do about it. `name(i)`
names variable `i`; `words` holds the vocabulary of the problem's form, as in
`_ISING_WORDS`.
"""
function _flip_message(groups::Vector{Vector{Int}}, n::Int, name, words)
    io = IOBuffer()
    firsts = [g[1] for g in groups]
    if length(groups) == 1 && length(groups[1]) == n
        print(io, "The objective does not change when every variable is flipped ($(words.flip)). ")
        print(io, "MFAOA starts all variables at $(words.start), where such an objective ")
        print(io, "has zero slope in every variable, so no variable would ever move. ")
        print(io, "Fix one variable to $(words.values)$(words.fix), for example $(name(firsts[1])). ")
        print(io, "This loses nothing, since flipping every variable does not change the objective value.")
    elseif length(groups) == 1
        print(io, "The objective does not change when the variables ")
        _print_names(io, name, groups[1])
        print(io, " are all flipped ($(words.flip)), ")
        print(io, "and no term of the objective links them to the other variables. ")
        print(io, "MFAOA starts all variables at $(words.start), where the objective ")
        print(io, "has zero slope in these variables, so they would never move. ")
        print(io, "Fix one of them to $(words.values)$(words.fix), for example $(name(firsts[1])). ")
        print(io, "This loses nothing, since flipping them does not change the objective value.")
    else
        print(io, "The objective does not change when all variables of any one of these ")
        print(io, "$(length(groups)) groups are flipped ($(words.flip)): ")
        for k in 1:min(length(groups), 3)
            k > 1 && print(io, k == length(groups) ? " and " : ", ")
            _print_set(io, name, groups[k])
        end
        length(groups) > 3 && print(io, " and $(length(groups) - 3) more groups")
        print(io, ". No term of the objective links a group to variables outside it. ")
        print(io, "MFAOA starts all variables at $(words.start), where the objective ")
        print(io, "has zero slope in these variables, so they would never move. ")
        print(io, "Fix one variable of each group to $(words.values)$(words.fix), for example ")
        _print_names(io, name, firsts)
        print(io, ". This loses nothing, since flipping a group does not change the objective value.")
    end
    String(take!(io))
end

"""
    _ising_name(i)

How messages name variable `i` of an [`IsingProblem`](@ref).
"""
_ising_name(i::Int) = "σ[$i]"

"""
    _qubo_name(i)

How messages name variable `i` of a [`QUBOProblem`](@ref).
"""
_qubo_name(i::Int) = "x[$i]"
