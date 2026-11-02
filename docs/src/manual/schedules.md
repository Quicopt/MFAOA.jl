# Schedules and the field scale

## The algorithm

MFAOA works on the [Ising form](@ref "The Ising form") of a problem, in
variables `σ[i]` that are -1 or +1. It represents each variable by an arrow
of length 1 in three dimensions. The height of the arrow, its z component,
is the variable's current value, between -1 and +1. Every arrow starts
horizontal, pointing along the x axis, so every value starts at 0.

MFAOA then runs a fixed number of steps. Step `k` turns every arrow twice:

1. about the vertical z axis, by an angle proportional to `γ[k]` and to the
   slope of the objective in that variable;
2. about the x axis, by the angle `2β[k]`.

After the last step, each variable is rounded to the sign of its arrow's
height. The two sequences of angles, `β` and `γ`, form the schedule.

## The linear schedule

Unless you pass a schedule of your own, MFAOA uses the linear schedule
`linear_schedule(p, τ)` with `p = 1000` steps and step size `τ = 0.5`:

```math
\gamma_k = \tau \, \frac{k}{p}, \qquad
\beta_k = \tau \left(1 - \frac{k - 1}{p}\right), \qquad k = 1, \dots, p .
```

With four steps, for example, `γ` grows and `β` shrinks like this:

```@example schedules
using MFAOA

schedule = linear_schedule(4, 0.5)
println("β: ", schedule.β)
println("γ: ", schedule.γ)
```

`solve(prob; p, τ)` sets the two numbers, and so do the attributes `"p"` and
`"tau"` in JuMP. A step size between 0.25 and 0.5 suits most problems.

## Number of steps

The time a solve takes grows in proportion to the number of steps, and the
solution usually improves with it. On a random problem with 2000
variables:

```@example schedules
using Random, SparseArrays, LinearAlgebra, Printf

# A random problem in Ising form: 2000 variables, each coupled to about six
# others with Gaussian weights, and weak random linear terms.
rng = MersenneTwister(1)
upper = triu(sprandn(rng, 2000, 2000, 6 / 2000), 1)
J = upper + upper'
h = 0.1 .* randn(rng, 2000)
problem = IsingProblem(h, J)

solve(problem; p = 1)    # compile the solver before timing it
@printf("%6s %12s %14s %9s\n", "p", "objective", "objective_mf", "time (s)")
for p in (10, 100, 1000, 10000)
    time = @elapsed result = solve(problem; p)
    @printf("%6d %12.2f %14.2f %9.4f\n", p, result.objective, result.objective_mf, time)
end
```

`objective_mf` is the objective at the values just before rounding. When it
is far from `objective`, many values were still far from -1 and +1 at the
end of the run, and more steps are likely to give a better solution.

## Field scale

In step `k`, the first turn of arrow `i` has the angle `2γ[k] m[i] / s`.
The number `m[i]` is the slope of the objective in variable `i`. The number
`s` is the problem's field scale, `fieldscale(problem)`, the typical size of
such a slope: the root mean square of `m[i]` over all variables and over
random values ±1 of the others. Because the slope is divided by `s`,
multiplying the objective by a constant changes neither the run nor the
solution:

```@example schedules
scaled = IsingProblem(1000 .* h, 1000 .* J)
println("field scale:         ", fieldscale(problem))
println("field scale, scaled: ", fieldscale(scaled))
println("same solution:       ", solve(scaled).σ == solve(problem).σ)
```

One step size `τ` therefore suits problems of any size and in any units. To measure the
slope in the units of the objective itself, build the problem with
`scale = 1`; the angles of the schedule then have to suit those units.

## Custom schedules

Any two vectors of equal length make a schedule. Pass them as
`Schedule(β, γ)` to `solve`, or set them in JuMP with
`set_attribute(model, "beta", β)` and `set_attribute(model, "gamma", γ)`.
In the following schedule, `γ` grows quadratically instead of linearly:

```@example schedules
p = 1000
β = [0.5 * (1 - (k - 1) / p) for k in 1:p]    # as in linear_schedule
γ = [0.5 * (k / p)^2 for k in 1:p]
result = solve(problem, Schedule(β, γ))
@printf("objective: %.2f\n", result.objective)
```

`IsingProblem` also accepts the keyword `Δ`: one positive factor per
variable on the angles `β`. The default, 1 for every variable, rarely needs
changing.
