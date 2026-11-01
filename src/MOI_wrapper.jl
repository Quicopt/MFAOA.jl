# MOI_wrapper.jl — MFAOA as a MathOptInterface optimizer, for use from JuMP.
#
# The optimizer accepts exactly the problem class the algorithm solves: binary
# variables, a linear or quadratic objective in either sense, and no
# constraints other than those on single variables that keep each variable
# binary. Those are the binary and integer types, fixing, and bounds. Bounds
# that still allow 0 and 1 are redundant and ignored. A bound that allows only
# one of them holds the variable there, like fixing it. Held variables are
# substituted out before the solve, which is also how a user breaks the flip
# symmetry of a problem without local fields (fix one variable).
#
# Any other constraint is rejected at copy time with MOI's
# UnsupportedConstraint error, which JuMP reports as "constraints of type …
# are not supported". A model that is accepted but cannot be solved (a
# continuous variable, a variable fixed to 0.5, a flip-symmetric objective)
# ends with termination status INVALID_MODEL, and one whose bounds allow a
# variable neither 0 nor 1 with INFEASIBLE, each with the reason in
# RawStatusString, as MOI prescribes.
#
# The model is not incremental: JuMP keeps it in a cache and hands it over
# whole through copy_to when optimize! is called.

const _SQF = MOI.ScalarQuadraticFunction{Float64}
const _SAF = MOI.ScalarAffineFunction{Float64}
const _VI = MOI.VariableIndex
const _RAW_ATTRIBUTES = ("p", "tau", "beta", "gamma")

"""
    MFAOA.Optimizer(; p = 1000, tau = 0.5)

The MFAOA solver for JuMP (as a MathOptInterface optimizer):

```julia
using JuMP, MFAOA
model = Model(MFAOA.Optimizer)
@variable(model, x[1:3], Bin)
@objective(model, Min, 2x[1]*x[2] - 3x[2]*x[3] - x[1] - x[2] + x[3])
optimize!(model)
value.(x), objective_value(model)
```

The model may contain binary variables, a linear or quadratic objective in
either sense, and fixed variables (`fix(x[i], 1)`). Bounds on a variable are
accepted when they allow 0 and 1, as in `0 <= x <= 1`, and integer variables
when their bounds allow only 0 and 1. A bound that allows only one of the two
values holds the variable there, as `fix` would. Nothing else is accepted.
Options,
set with `set_attribute(model, name, value)`:

- `"p"`: the number of steps of the linear schedule (default 1000);
- `"tau"`: its step size (default 0.5);
- `"beta"`, `"gamma"`: an explicit [`Schedule`](@ref) as two vectors of equal
  length, used instead of the linear schedule when both are set.

A solve ends with `LOCALLY_SOLVED`: the solver returns a solution, without
a proof that no better one exists. `INVALID_MODEL` means that MFAOA cannot
solve the model, and `raw_status(model)` says why and what to do. Most
often, the objective does not change when every variable `x` is replaced by
`1 - x`, as in Max-Cut; fixing one variable, say with `fix(x[1], 0)`, solves
that.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    # options
    p::Int
    tau::Float64
    beta::Union{Nothing,Vector{Float64}}
    gamma::Union{Nothing,Vector{Float64}}
    silent::Bool
    # the model, as handed over by copy_to (objective in the user's sense)
    has_model::Bool
    n::Int
    sense::MOI.OptimizationSense
    Q::SparseMatrixCSC{Float64,Int}  # objective x'Qx + c'x + constant
    c::Vector{Float64}
    constant::Float64
    constraint_types::Vector{Tuple{Type,Type}}   # as listed by the source model
    isbinary::Vector{Bool}
    isinteger::Vector{Bool}
    fixed::Vector{Float64}          # NaN where the variable is not fixed
    lower::Vector{Float64}          # bounds, -Inf and Inf where there are none
    upper::Vector{Float64}
    names::Vector{String}           # "" where the model gives no name
    # results
    status::MOI.TerminationStatusCode
    raw_status::String
    x::Vector{Float64}
    objective_value::Float64
    solve_time::Float64
end

function Optimizer(; p::Integer = 1000, tau::Real = 0.5)
    o = Optimizer(p, tau, nothing, nothing, false,
                  false, 0, MOI.FEASIBILITY_SENSE, spzeros(0, 0), Float64[], 0.0,
                  Tuple{Type,Type}[], Bool[], Bool[], Float64[], Float64[], Float64[],
                  String[], MOI.OPTIMIZE_NOT_CALLED, "", Float64[], NaN, 0.0)
    MOI.set(o, MOI.RawOptimizerAttribute("p"), p)
    MOI.set(o, MOI.RawOptimizerAttribute("tau"), tau)
    o
end

Base.show(io::IO, ::Optimizer) = print(io, "MFAOA.Optimizer")

MOI.get(::Optimizer, ::MOI.SolverName) = "MFAOA"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(@__MODULE__))

function MOI.empty!(o::Optimizer)
    o.has_model = false
    o.n = 0
    o.sense = MOI.FEASIBILITY_SENSE
    o.Q = spzeros(0, 0)
    o.c = Float64[]
    o.constant = 0.0
    o.constraint_types = Tuple{Type,Type}[]
    o.isbinary = Bool[]
    o.isinteger = Bool[]
    o.fixed = Float64[]
    o.lower = Float64[]
    o.upper = Float64[]
    o.names = String[]
    _reset_results!(o)
    o
end

function _reset_results!(o::Optimizer)
    o.status = MOI.OPTIMIZE_NOT_CALLED
    o.raw_status = ""
    o.x = Float64[]
    o.objective_value = NaN
    o.solve_time = 0.0
    o
end

MOI.is_empty(o::Optimizer) = !o.has_model

MOI.supports_incremental_interface(::Optimizer) = false

# --- options -----------------------------------------------------------------

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(o::Optimizer, ::MOI.Silent, value::Bool) = (o.silent = value; nothing)
MOI.get(o::Optimizer, ::MOI.Silent) = o.silent

MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute) = attr.name in _RAW_ATTRIBUTES

function MOI.set(o::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    if attr.name == "p"
        value isa Integer && value ≥ 1 ||
            throw(ArgumentError("\"p\" must be a positive integer, got $value"))
        o.p = value
    elseif attr.name == "tau"
        value isa Real && value > 0 ||
            throw(ArgumentError("\"tau\" must be a positive number, got $value"))
        o.tau = value
    elseif attr.name == "beta"
        o.beta = value === nothing ? nothing : Vector{Float64}(value)
    elseif attr.name == "gamma"
        o.gamma = value === nothing ? nothing : Vector{Float64}(value)
    else
        throw(MOI.UnsupportedAttribute(attr))
    end
    nothing
end

function MOI.get(o::Optimizer, attr::MOI.RawOptimizerAttribute)
    attr.name == "p" && return o.p
    attr.name == "tau" && return o.tau
    attr.name == "beta" && return o.beta
    attr.name == "gamma" && return o.gamma
    throw(MOI.UnsupportedAttribute(attr))
end

function _schedule(o::Optimizer)
    if o.beta === nothing && o.gamma === nothing
        return linear_schedule(o.p, o.tau)
    elseif o.beta === nothing || o.gamma === nothing
        throw(ArgumentError("set both \"beta\" and \"gamma\" for an explicit schedule, or neither for the linear one"))
    end
    Schedule(o.beta, o.gamma)
end

# --- what the model may contain ----------------------------------------------

# Constraints on single variables, which keep the variables binary or hold
# them at 0 or 1; optimize! checks which values each variable may take.
const _VARIABLE_SETS = Union{MOI.ZeroOne,MOI.Integer,MOI.EqualTo{Float64},MOI.GreaterThan{Float64},
                             MOI.LessThan{Float64},MOI.Interval{Float64}}
MOI.supports_constraint(::Optimizer, ::Type{_VI}, ::Type{<:_VARIABLE_SETS}) = true

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{F}) where {F<:Union{_VI,_SAF,_SQF}} = true
# Variable names are kept for the messages in RawStatusString; other names are
# accepted and ignored. Any other variable or constraint attribute is rejected
# by copy_to.
MOI.supports(::Optimizer, ::MOI.Name) = true
MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{_VI}) = true
MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{<:MOI.ConstraintIndex}) = true

MOI.get(o::Optimizer, ::MOI.ObjectiveSense) = o.sense
MOI.get(o::Optimizer, ::MOI.NumberOfVariables) = o.n
MOI.get(o::Optimizer, ::MOI.ListOfVariableIndices) = [_VI(i) for i in 1:o.n]
MOI.get(o::Optimizer, ::MOI.ListOfConstraintTypesPresent) = copy(o.constraint_types)

# --- copy_to: take the whole model over from JuMP's cache --------------------

"""
    _check_supported(dest::Optimizer, src::MOI.ModelLike)

Throw MOI's `UnsupportedConstraint` or `UnsupportedAttribute` error for the
first constraint type or attribute in `src` that `dest` does not accept.
"""
function _check_supported(dest::Optimizer, src::MOI.ModelLike)
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(dest, F, S) || throw(MOI.UnsupportedConstraint{F,S}())
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            MOI.supports(dest, attr, MOI.ConstraintIndex{F,S}) || throw(MOI.UnsupportedAttribute(attr))
        end
    end
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        MOI.supports(dest, attr) || throw(MOI.UnsupportedAttribute(attr))
    end
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        MOI.supports(dest, attr, _VI) || throw(MOI.UnsupportedAttribute(attr))
    end
    nothing
end

"""
    _objective_data(src, index_map, n) -> (Q, c, constant)

The objective of `src` as `x' * Q * x + c' * x + constant` over the `n`
variables numbered by `index_map`, with `Q` sparse and in the user's sense.
"""
function _objective_data(src::MOI.ModelLike, index_map, n::Int)
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    f = convert(_SQF, MOI.Utilities.map_indices(index_map, MOI.get(src, MOI.ObjectiveFunction{F}())))
    c = zeros(n)
    for t in f.affine_terms
        c[t.variable.value] += t.coefficient
    end
    # MOI stores ½ x'Qx: a term (coef, i, j) means coef·xᵢxⱼ for i ≠ j and
    # (coef/2)·xᵢ² for i == j. `sparse` adds up repeated pairs.
    terms = f.quadratic_terms
    rows = [t.variable_1.value for t in terms]
    cols = [t.variable_2.value for t in terms]
    vals = [t.variable_1 == t.variable_2 ? t.coefficient / 2 : t.coefficient for t in terms]
    sparse(rows, cols, vals, n, n), c, f.constant
end

"""
    _variable_names(src, variables) -> Vector{String}

The names of `variables` in `src`, with `""` for a variable without one.
"""
function _variable_names(src::MOI.ModelLike, variables::Vector{_VI})
    MOI.VariableName() in MOI.get(src, MOI.ListOfVariableAttributesSet()) ||
        return fill("", length(variables))
    [MOI.get(src, MOI.VariableName(), vi) for vi in variables]
end

"""
    _variable_constraints(src, index_map, S) -> Vector{Tuple{Int,S}}

The constraints of `src` that put a single variable into a set of type `S`,
as pairs of the variable's number in `index_map` and the set. Records the
constraints' new indices in `index_map`.
"""
function _variable_constraints(src::MOI.ModelLike, index_map, ::Type{S}) where {S}
    constraints = Tuple{Int,S}[]
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{_VI,S}())
        i = index_map[MOI.get(src, MOI.ConstraintFunction(), ci)].value
        index_map[ci] = MOI.ConstraintIndex{_VI,S}(i)
        push!(constraints, (i, MOI.get(src, MOI.ConstraintSet(), ci)))
    end
    constraints
end

"""
    _copy_variable_constraints!(dest::Optimizer, src, index_map)

Copy the type, fixing and bounds of every variable of `src` into `dest`.
"""
function _copy_variable_constraints!(dest::Optimizer, src::MOI.ModelLike, index_map)
    n = dest.n
    dest.isbinary, dest.isinteger = falses(n), falses(n)
    dest.fixed, dest.lower, dest.upper = fill(NaN, n), fill(-Inf, n), fill(Inf, n)
    for (i, _) in _variable_constraints(src, index_map, MOI.ZeroOne)
        dest.isbinary[i] = true
    end
    for (i, _) in _variable_constraints(src, index_map, MOI.Integer)
        dest.isinteger[i] = true
    end
    for (i, set) in _variable_constraints(src, index_map, MOI.EqualTo{Float64})
        dest.fixed[i] = set.value
    end
    for (i, set) in _variable_constraints(src, index_map, MOI.GreaterThan{Float64})
        dest.lower[i] = max(dest.lower[i], set.lower)
    end
    for (i, set) in _variable_constraints(src, index_map, MOI.LessThan{Float64})
        dest.upper[i] = min(dest.upper[i], set.upper)
    end
    for (i, set) in _variable_constraints(src, index_map, MOI.Interval{Float64})
        dest.lower[i] = max(dest.lower[i], set.lower)
        dest.upper[i] = min(dest.upper[i], set.upper)
    end
    nothing
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    _check_supported(dest, src)
    dest.constraint_types = MOI.get(src, MOI.ListOfConstraintTypesPresent())
    variables = MOI.get(src, MOI.ListOfVariableIndices())
    n = length(variables)
    index_map = MOI.Utilities.IndexMap()
    for (i, vi) in enumerate(variables)
        index_map[vi] = _VI(i)
    end
    dest.n = n
    dest.names = _variable_names(src, variables)
    _copy_variable_constraints!(dest, src, index_map)
    dest.sense = MOI.get(src, MOI.ObjectiveSense())
    if dest.sense == MOI.FEASIBILITY_SENSE
        dest.Q, dest.c, dest.constant = spzeros(n, n), zeros(n), 0.0
    else
        dest.Q, dest.c, dest.constant = _objective_data(src, index_map, n)
    end
    dest.has_model = true
    index_map
end

# --- optimize! ---------------------------------------------------------------

"""
    _reject!(o::Optimizer, status, reason)

End the solve without a solution, with the termination status `status` and
`reason` as the raw status.
"""
function _reject!(o::Optimizer, status::MOI.TerminationStatusCode, reason::String)
    o.status = status
    o.raw_status = reason
    o.x = Float64[]
    o.objective_value = NaN
    nothing
end

"""
    _variable_name(o::Optimizer, i) -> String

The name of variable `i` in messages: its name in the model, such as `x[3]`,
or `variable 3` if it has none.
"""
_variable_name(o::Optimizer, i::Int) = isempty(o.names[i]) ? "variable $i" : o.names[i]

"""
    _isbinary(o::Optimizer, i) -> Bool

`true` when variable `i` can take no values other than 0 and 1: it is binary,
or it is an integer variable with bounds strictly between -1 and 2.
"""
_isbinary(o::Optimizer, i::Int) =
    o.isbinary[i] || (o.isinteger[i] && o.lower[i] > -1 && o.upper[i] < 2)

"""
    _held_value(o::Optimizer, i) -> Union{Float64,Tuple{MOI.TerminationStatusCode,String}}

The value at which variable `i` is held: 0 or 1 when it is fixed, or when its
bounds allow only that value, and `NaN` when it is free to be 0 or 1. For a
variable that rules out a solve, the termination status and the reason
instead: `INVALID_MODEL` when the variable may take other values than 0 and 1,
`INFEASIBLE` when its fixing or bounds leave it no value.
"""
function _held_value(o::Optimizer, i::Int)
    name = _variable_name(o, i)
    lower, upper = o.lower[i], o.upper[i]
    if !isnan(o.fixed[i])
        value = o.fixed[i]
        value == 0 || value == 1 ||
            return (MOI.INVALID_MODEL, "$name is fixed to $value; MFAOA accepts variables fixed to 0 or 1 only.")
        lower ≤ value ≤ upper ||
            return (MOI.INFEASIBLE, "$name is fixed to $value, but its bounds require $lower ≤ $name ≤ $upper.")
        return value
    end
    _isbinary(o, i) ||
        return (MOI.INVALID_MODEL, "$name is neither binary nor fixed; MFAOA solves problems in binary variables only.")
    allows_0, allows_1 = lower ≤ 0 ≤ upper, lower ≤ 1 ≤ upper
    allows_0 && allows_1 && return NaN
    allows_0 && return 0.0
    allows_1 && return 1.0
    (MOI.INFEASIBLE, "The bounds $lower ≤ $name ≤ $upper allow neither 0 nor 1.")
end

# The words in which `_flip_message` explains the problem to a JuMP user.
const _JUMP_WORDS = (flip = "x → 1 - x", start = "1/2", values = "0 or 1", fix = " with JuMP's `fix`")

"""
    _flip_status(o::Optimizer, prob, free) -> Union{String,Nothing}

The raw status for a model that MFAOA cannot solve because of a group of
variables it cannot move (see [`isflipsymmetric`](@ref)), or `nothing` if
there is no such group. `free` are the model's free variables, and `prob` is
the objective in them.
"""
function _flip_status(o::Optimizer, prob::IsingProblem, free::Vector{Int})
    groups = _flip_groups(prob)
    isempty(groups) && return nothing
    _flip_message(groups, length(free), k -> _variable_name(o, free[k]), _JUMP_WORDS)
end

"""
    _objective_value(o::Optimizer, x)

`x' * Q * x + c' * x + constant` for a full assignment `x`, in the user's sense.
"""
_objective_value(o::Optimizer, x::Vector{Float64}) = dot(x, o.Q, x) + dot(o.c, x) + o.constant

# The sparse product beats a dense one (OpenBLAS) below about 25-30% fill,
# measured on an Apple M4 for n = 500 to 2000.
const _DENSE_FROM = 0.25

"""
    _coupling_storage(Q::SparseMatrixCSC)

`Q` as the solve should store it: dense once the pairs it couples, the pattern
of `Q + Qᵀ`, fill `_DENSE_FROM` of the matrix; sparse below that.
"""
function _coupling_storage(Q::SparseMatrixCSC)
    n = size(Q, 1)
    nnz(Q + transpose(Q)) ≥ _DENSE_FROM * n^2 ? Matrix(Q) : Q
end

"""
    _reduced_problem(o::Optimizer, x) -> IsingProblem

The objective as a function of the free variables alone, those with
`x[i] = NaN`, the others held at their values in `x`. With `f` the free and
`F` the held variables,

    x'Qx + c'x = x_f' Q_ff x_f + (c_f + Q_fF x_F + Q_Ffᵀ x_F)' x_f + x_F' Q_FF x_F + c_F' x_F,

so the held variables add to the linear term and to the constant.
"""
function _reduced_problem(o::Optimizer, x::Vector{Float64})
    free = findall(isnan, x)
    held = findall(!isnan, x)
    x_F = x[held]
    Q = o.Q
    c = o.c[free] + Q[free, held] * x_F + transpose(Q[held, free]) * x_F
    constant = o.constant + dot(o.c[held], x_F) + dot(x_F, Q[held, held], x_F)
    sense = o.sense == MOI.MAX_SENSE ? :max : :min
    IsingProblem(QUBOProblem(_coupling_storage(Q[free, free]); c, constant, sense))
end

"""
    _finish!(o::Optimizer, x, status, raw_status, start)

Record the solution `x` with its objective value, the status, and the time
since `start`.
"""
function _finish!(o::Optimizer, x::Vector{Float64}, status::MOI.TerminationStatusCode,
                  raw_status::String, start::Float64)
    o.x = x
    o.objective_value = _objective_value(o, x)
    o.status = status
    o.raw_status = raw_status
    o.solve_time = time() - start
    nothing
end

function MOI.optimize!(o::Optimizer)
    start = time()
    _reset_results!(o)
    x = Vector{Float64}(undef, o.n)
    for i in 1:o.n
        value = _held_value(o, i)
        value isa Float64 || return _reject!(o, value...)
        x[i] = value
    end
    free = findall(isnan, x)
    if o.sense == MOI.FEASIBILITY_SENSE || isempty(free)
        x[free] .= 0.0
        return _finish!(o, x, MOI.OPTIMAL,
                        isempty(free) ? "all variables fixed" : "feasibility problem: any assignment is a solution",
                        start)
    end
    prob = _reduced_problem(o, x)
    reason = _flip_status(o, prob, free)
    reason === nothing || return _reject!(o, MOI.INVALID_MODEL, reason)
    sched = _schedule(o)
    x[free] .= solve(prob, sched).x
    _finish!(o, x, MOI.LOCALLY_SOLVED,
             "MFAOA ran $(length(sched)) steps on $(length(free)) free variables.", start)
end

# --- results -----------------------------------------------------------------

MOI.get(o::Optimizer, ::MOI.TerminationStatus) = o.status
MOI.get(o::Optimizer, ::MOI.RawStatusString) = o.raw_status
MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.solve_time
MOI.get(o::Optimizer, ::MOI.ResultCount) =
    o.status == MOI.OPTIMAL || o.status == MOI.LOCALLY_SOLVED ? 1 : 0

function MOI.get(o::Optimizer, attr::MOI.PrimalStatus)
    attr.result_index == 1 && MOI.get(o, MOI.ResultCount()) == 1 || return MOI.NO_SOLUTION
    MOI.FEASIBLE_POINT
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(o::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(o, attr)
    o.objective_value
end

function MOI.get(o::Optimizer, attr::MOI.VariablePrimal, vi::_VI)
    MOI.check_result_index_bounds(o, attr)
    MOI.throw_if_not_valid(o, vi)
    o.x[vi.value]
end

MOI.is_valid(o::Optimizer, vi::_VI) = 1 ≤ vi.value ≤ o.n
