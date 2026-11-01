# MOI_wrapper.jl — MFAOA as a MathOptInterface optimizer, which is what JuMP
# talks to.
#
# The optimizer accepts exactly the problem class the algorithm solves: binary
# variables, a linear or quadratic objective in either sense, and no
# constraints other than fixing a variable to 0 or 1. Fixed variables are
# substituted out before the solve, which is also how a user breaks the
# flip symmetry of a problem without local fields (fix one variable).
#
# Any other constraint is rejected at copy time with MOI's
# UnsupportedConstraint error, which JuMP reports as "constraints of type …
# are not supported". A model that is accepted but cannot be solved (a
# continuous variable, a variable fixed to 0.5, a flip-symmetric objective)
# ends with termination status INVALID_MODEL and the reason in
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
either sense, and fixed variables (`fix(x[i], 1)`); nothing else. Options,
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
    isbinary::Vector{Bool}
    fixed::Vector{Float64}          # NaN where the variable is free
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
                  false, 0, MOI.FEASIBILITY_SENSE, spzeros(0, 0), Float64[], 0.0, Bool[], Float64[],
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
    o.isbinary = Bool[]
    o.fixed = Float64[]
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

MOI.supports_constraint(::Optimizer, ::Type{_VI}, ::Type{MOI.ZeroOne}) = true
MOI.supports_constraint(::Optimizer, ::Type{_VI}, ::Type{MOI.EqualTo{Float64}}) = true

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
MOI.get(o::Optimizer, ::MOI.ListOfConstraintTypesPresent) = begin
    types = Tuple{Type,Type}[]
    any(o.isbinary) && push!(types, (_VI, MOI.ZeroOne))
    any(!isnan, o.fixed) && push!(types, (_VI, MOI.EqualTo{Float64}))
    types
end

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

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    _check_supported(dest, src)
    variables = MOI.get(src, MOI.ListOfVariableIndices())
    n = length(variables)
    index_map = MOI.Utilities.IndexMap()
    for (i, vi) in enumerate(variables)
        index_map[vi] = _VI(i)
    end
    dest.n = n
    dest.isbinary = falses(n)
    dest.fixed = fill(NaN, n)
    dest.names = _variable_names(src, variables)
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{_VI,MOI.ZeroOne}())
        i = index_map[MOI.get(src, MOI.ConstraintFunction(), ci)].value
        dest.isbinary[i] = true
        index_map[ci] = MOI.ConstraintIndex{_VI,MOI.ZeroOne}(i)
    end
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{_VI,MOI.EqualTo{Float64}}())
        i = index_map[MOI.get(src, MOI.ConstraintFunction(), ci)].value
        dest.fixed[i] = MOI.get(src, MOI.ConstraintSet(), ci).value
        index_map[ci] = MOI.ConstraintIndex{_VI,MOI.EqualTo{Float64}}(i)
    end
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
    _invalid!(o::Optimizer, reason)

End the solve with `INVALID_MODEL`, giving `reason` as the raw status.
"""
function _invalid!(o::Optimizer, reason::String)
    o.status = MOI.INVALID_MODEL
    o.raw_status = reason
    o.x = Float64[]
    o.objective_value = NaN
    nothing
end

"""
    _variable_name(o::Optimizer, i) -> String

How messages name variable `i`: by its name in the model, such as `x[3]`, or
as `variable 3` if it has none.
"""
_variable_name(o::Optimizer, i::Int) = isempty(o.names[i]) ? "variable $i" : o.names[i]

"""
    _invalid_variable(o::Optimizer) -> Union{String,Nothing}

Why the model's variables put it outside the problem class (a variable that is
neither binary nor fixed, or one fixed to a value other than 0 or 1), or
`nothing` if they don't.
"""
function _invalid_variable(o::Optimizer)
    for i in 1:o.n
        name = _variable_name(o, i)
        if isnan(o.fixed[i])
            o.isbinary[i] ||
                return "$name is neither binary nor fixed; MFAOA solves problems in binary variables only."
        elseif !(o.fixed[i] == 0 || o.fixed[i] == 1)
            return "$name is fixed to $(o.fixed[i]); MFAOA accepts variables fixed to 0 or 1 only."
        end
    end
    nothing
end

# The words in which `_flip_message` explains the problem to a JuMP user.
const _JUMP_WORDS = (flip = "x → 1 - x", start = "1/2", values = "0 or 1", fix = " with JuMP's `fix`")

"""
    _flip_status(o::Optimizer, prob, free) -> Union{String,Nothing}

Why MFAOA cannot solve the model with the free variables `free`, whose
objective is `prob`, because of a group of variables it cannot move (see
[`isflipsymmetric`](@ref)); `nothing` if there is no such group.
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
    _reduced_problem(o::Optimizer, free) -> IsingProblem

The objective as a function of the variables `free` alone, the others held at
their fixed values. With `f` the free and `F` the fixed variables,

    x'Qx + c'x = x_f' Q_ff x_f + (c_f + Q_fF x_F + Q_Ffᵀ x_F)' x_f + x_F' Q_FF x_F + c_F' x_F,

so the fixed variables add to the linear term and to the constant.
"""
function _reduced_problem(o::Optimizer, free::Vector{Int})
    fixed = findall(!isnan, o.fixed)
    x_F = o.fixed[fixed]
    Q = o.Q
    c = o.c[free] + Q[free, fixed] * x_F + transpose(Q[fixed, free]) * x_F
    constant = o.constant + dot(o.c[fixed], x_F) + dot(x_F, Q[fixed, fixed], x_F)
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
    reason = _invalid_variable(o)
    reason === nothing || return _invalid!(o, reason)
    free = findall(isnan, o.fixed)
    x = copy(o.fixed)
    if o.sense == MOI.FEASIBILITY_SENSE || isempty(free)
        x[free] .= 0.0
        return _finish!(o, x, MOI.OPTIMAL,
                        isempty(free) ? "all variables fixed" : "feasibility problem: any assignment is a solution",
                        start)
    end
    prob = _reduced_problem(o, free)
    reason = _flip_status(o, prob, free)
    reason === nothing || return _invalid!(o, reason)
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
