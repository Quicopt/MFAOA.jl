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

A MathOptInterface optimizer running the mean-field AOA, for use from JuMP:

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

- `"p"`: number of layers of the linear schedule (default 1000);
- `"tau"`: its time step (default 0.5);
- `"beta"`, `"gamma"`: an explicit [`Schedule`](@ref) as two vectors of equal
  length, used instead of the linear schedule when both are set.

A solve ends with `LOCALLY_SOLVED`: the algorithm returns a good configuration
without a certificate of optimality. `INVALID_MODEL` with an explanation in
`raw_status(model)` means the model is outside the problem class, most often
because its objective is invariant under flipping all variables
(`x → 1 - x`, as in max-cut); fix one variable to break that symmetry.
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
    Q::Matrix{Float64}
    c::Vector{Float64}
    constant::Float64
    isbinary::Vector{Bool}
    fixed::Vector{Float64}          # NaN where the variable is free
    # results
    status::MOI.TerminationStatusCode
    raw_status::String
    x::Vector{Float64}
    objective_value::Float64
    solve_time::Float64
end

function Optimizer(; p::Integer = 1000, tau::Real = 0.5)
    o = Optimizer(p, tau, nothing, nothing, false,
                  false, 0, MOI.FEASIBILITY_SENSE, zeros(0, 0), Float64[], 0.0, Bool[], Float64[],
                  MOI.OPTIMIZE_NOT_CALLED, "", Float64[], NaN, 0.0)
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
    o.Q = zeros(0, 0)
    o.c = Float64[]
    o.constant = 0.0
    o.isbinary = Bool[]
    o.fixed = Float64[]
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
# Names are accepted and ignored (JuMP keeps them in its own cache). Any other
# variable or constraint attribute is rejected by copy_to.
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

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
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

    variables = MOI.get(src, MOI.ListOfVariableIndices())
    n = length(variables)
    index_map = MOI.Utilities.IndexMap()
    for (i, vi) in enumerate(variables)
        index_map[vi] = _VI(i)
    end
    dest.n = n
    dest.isbinary = falses(n)
    dest.fixed = fill(NaN, n)
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
    dest.Q = zeros(n, n)
    dest.c = zeros(n)
    dest.constant = 0.0
    if dest.sense != MOI.FEASIBILITY_SENSE
        F = MOI.get(src, MOI.ObjectiveFunctionType())
        f = MOI.Utilities.map_indices(index_map, MOI.get(src, MOI.ObjectiveFunction{F}()))
        q = convert(_SQF, f)
        dest.constant = q.constant
        for t in q.affine_terms
            dest.c[t.variable.value] += t.coefficient
        end
        # MOI stores ½ x'Qx: a term (coef, i, j) means coef·xᵢxⱼ for i ≠ j and
        # (coef/2)·xᵢ² for i == j.
        for t in q.quadratic_terms
            i = t.variable_1.value
            j = t.variable_2.value
            dest.Q[i, j] += i == j ? t.coefficient / 2 : t.coefficient
        end
    end
    dest.has_model = true
    index_map
end

# --- optimize! ---------------------------------------------------------------

function _invalid!(o::Optimizer, reason::String)
    o.status = MOI.INVALID_MODEL
    o.raw_status = reason
    o.x = Float64[]
    o.objective_value = NaN
    nothing
end

# x' Q x + c' x + constant for a full assignment, in the user's sense.
function _objective_value(o::Optimizer, x::Vector{Float64})
    v = o.constant
    @inbounds for j in 1:o.n
        xj = x[j]
        iszero(xj) && continue
        acc = o.c[j]
        for i in 1:o.n
            acc += o.Q[i, j] * x[i]
        end
        v += xj * acc
    end
    v
end

function MOI.optimize!(o::Optimizer)
    start = time()
    _reset_results!(o)
    for i in 1:o.n
        if isnan(o.fixed[i])
            o.isbinary[i] || return _invalid!(o,
                "variable $i is neither binary nor fixed; MFAOA solves problems in binary variables only")
        elseif !(o.fixed[i] == 0 || o.fixed[i] == 1)
            return _invalid!(o, "variable $i is fixed to $(o.fixed[i]), which is not 0 or 1")
        end
    end
    free = findall(isnan, o.fixed)
    x = copy(o.fixed)
    if o.sense == MOI.FEASIBILITY_SENSE || isempty(free)
        x[free] .= 0.0
        o.x = x
        o.objective_value = _objective_value(o, x)
        o.status = MOI.OPTIMAL
        o.raw_status = isempty(free) ? "all variables fixed" : "feasibility problem: any assignment is a solution"
        o.solve_time = time() - start
        return nothing
    end
    # Substitute the fixed variables out: their couplings to free variables
    # become linear terms, their mutual couplings a constant.
    Q = o.Q[free, free]
    c = o.c[free]
    constant = o.constant
    @inbounds for j in 1:o.n
        isnan(o.fixed[j]) && continue
        vj = o.fixed[j]
        iszero(vj) && continue
        constant += o.c[j] * vj
        for (a, i) in enumerate(free)
            c[a] += (o.Q[i, j] + o.Q[j, i]) * vj
        end
        for i in 1:o.n
            isnan(o.fixed[i]) && continue
            constant += o.Q[i, j] * o.fixed[i] * vj
        end
    end
    prob = IsingProblem(QUBOProblem(Q; c = c, constant = constant,
                                    sense = o.sense == MOI.MAX_SENSE ? :max : :min))
    if isflipsymmetric(prob)
        return _invalid!(o,
            "the objective is invariant under flipping all free variables (x → 1 - x), " *
            "so the mean-field dynamics cannot choose a side; fix one variable to break the symmetry")
    end
    res = solve(prob, _schedule(o))
    x[free] .= res.x
    o.x = x
    o.objective_value = res.objective
    o.status = MOI.LOCALLY_SOLVED
    o.raw_status = "mean-field AOA finished: $(length(free)) free variables, $(length(_schedule(o))) layers"
    o.solve_time = time() - start
    nothing
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
