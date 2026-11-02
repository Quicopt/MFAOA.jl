# Solving from JuMP

To solve a JuMP model with MFAOA, create it with `Model(MFAOA.Optimizer)`.

## Supported models

A model may contain

- binary variables (`Bin`), with or without bounds such as `0 <= x <= 1`,
- integer variables (`Int`) whose bounds allow only 0 and 1,
- a linear or quadratic objective, to be minimized or maximized,
- variables fixed to 0 or 1 with `fix`.

A bound that allows only one of the values 0 and 1, such as `x >= 1` on a
binary variable, holds the variable at that value, just as `fix` does.

As an example, choose which of four features to build. Each feature has a
value. Some pairs of features interact: features 1 and 3 overlap, so
building both loses 5, while features 2 and 4 together are worth 2 more.

```@example jump
using JuMP, MFAOA

model = Model(MFAOA.Optimizer)
@variable(model, build[1:4], Bin)    # build[f] = 1: feature f is built
@objective(model, Max,
    3build[1] + 2build[2] + 4build[3] + build[4]
    + build[1] * build[2] - 5build[1] * build[3]
    + 2build[2] * build[4] - build[3] * build[4])
optimize!(model)

println("status: ", termination_status(model))
println("build:  ", round.(Int, value.(build)))
println("value:  ", objective_value(model))
```

The best choice builds features 1, 2 and 4, for a value of 9. If feature 3
must be built, fix its variable to 1 and solve again:

```@example jump
fix(build[3], 1)
optimize!(model)

println("build:  ", round.(Int, value.(build)))
println("value:  ", objective_value(model))
```

`unfix(build[3])` removes the fixing again.

## Unsupported models

MFAOA solves problems without constraints, so JuMP rejects any constraint
other than those listed above, with an error:

```@example jump
model = Model(MFAOA.Optimizer)
@variable(model, x[1:2], Bin)
try
    @constraint(model, x[1] + x[2] <= 1)
catch err
    println(first(split(sprint(showerror, err), "\n")))
end
```

Other models reach `optimize!` and end with the status `INVALID_MODEL`. Then
`raw_status` says why:

```@example jump
model = Model(MFAOA.Optimizer)
@variable(model, x, Bin)
@variable(model, y)                  # a continuous variable
@objective(model, Min, x * y - x)
optimize!(model)

println(termination_status(model))
println(raw_status(model))
```

A model also ends with `INVALID_MODEL` if an integer variable may take other
values than 0 and 1, if a variable is fixed to a value other than 0 or 1,
or if the objective does not change when every variable `x` is replaced by
`1 - x`. Max-Cut has such an objective, and
[Symmetric problems and fixing variables](@ref) explains what to do about it.

## Statuses and results

| `termination_status` | Meaning |
|:--- |:--- |
| `LOCALLY_SOLVED` | MFAOA returned a solution, without a proof that no better one exists. |
| `OPTIMAL` | Every variable is fixed, or the model has no objective, so there was nothing to search. |
| `INVALID_MODEL` | MFAOA cannot solve the model; `raw_status` says why. |
| `INFEASIBLE` | The bounds of a variable allow neither 0 nor 1. |

After `LOCALLY_SOLVED` or `OPTIMAL`, `value`, `objective_value` and
`solve_time` give the solution, its objective value and the time the solve
took, and `raw_status` gives a summary of the run.

## Options

Set options with `set_attribute`:

| Attribute | Meaning | Default |
|:--- |:--- |:--- |
| `"p"` | the number of steps that MFAOA runs | 1000 |
| `"tau"` | the step size `τ` of the linear schedule | 0.5 |
| `"beta"`, `"gamma"` | a schedule of your own, as two vectors of equal length; used instead of the linear schedule when both are set | not set |

[Schedules and the field scale](@ref) explains what these settings do. More
steps usually give a better solution and take proportionally longer:

```@example jump
model = Model(MFAOA.Optimizer)
@variable(model, build[1:4], Bin)
@objective(model, Max,
    3build[1] + 2build[2] + 4build[3] + build[4]
    + build[1] * build[2] - 5build[1] * build[3]
    + 2build[2] * build[4] - build[3] * build[4])
set_attribute(model, "p", 200)
optimize!(model)

println(raw_status(model))
```

## Storage of the objective

MFAOA stores the quadratic part of the objective as a sparse matrix, unless
at least a quarter of all pairs of variables appear in it together. Then it
stores a dense matrix. [Performance](@ref) explains why this matters for the
run time.
