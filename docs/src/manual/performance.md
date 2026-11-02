# Performance

## Cost of a solve

Each step multiplies the coupling matrix `J` of the Ising form by a vector,
and then does a fixed amount of work per variable. For a dense matrix, the
product takes one multiplication per entry, `n²` in all; for a sparse
matrix, one per stored entry. The time of a solve grows in proportion to the
number of steps.

## Sparse and dense matrices

Most of the time goes into the product with `J`, so a sparse `J` pays off
when most of its entries are zero. The table compares the two for 100
steps on random problems with 2000 variables, in which a growing share of
all pairs of variables is coupled:

```@example performance
using MFAOA, Random, SparseArrays, LinearAlgebra, Printf

# A random problem in Ising form with n variables, in which about the share
# `fill` of all pairs is coupled, with Gaussian weights and linear terms.
function random_couplings(n, fill)
    rng = MersenneTwister(1)
    upper = triu(sprandn(rng, n, n, fill), 1)
    return randn(rng, n), upper + upper'
end

# The best of three timings of a solve with p steps.
best_time(problem, p) = minimum(@elapsed(solve(problem; p)) for _ in 1:3)

@printf("%14s %12s %10s %10s\n", "pairs coupled", "stored", "dense (s)", "sparse (s)")
for fill in (0.001, 0.01, 0.1, 0.3)
    h, J = random_couplings(2000, fill)
    sparse_problem = IsingProblem(h, J)
    dense_problem = IsingProblem(h, Matrix(J))
    @printf("%13.1f%% %12d %10.4f %10.4f\n", 100fill, nnz(J),
            best_time(dense_problem, 100), best_time(sparse_problem, 100))
end
```

The share at which sparse and dense take equally long depends on the
computer; the table shows it for the computer that built this page. `MFAOA.Optimizer` stores the objective as a dense
matrix once at least a quarter of all pairs of variables appear in it, and
as a sparse one below that. An `IsingProblem` or `QUBOProblem` uses the
matrix you give it.

The product with a dense matrix runs through the BLAS library that Julia's
`LinearAlgebra` uses. Loading a faster BLAS, such as MKL.jl or
AppleAccelerate.jl where it is available, speeds up dense problems without
changes to your code.

## Repeated solves

`solve` allocates memory for the run and for the `Result`. To solve many
problems of the same size, allocate that memory once as a `Workspace`, and
call `solve!`, which allocates nothing:

```@example performance
h, J = random_couplings(2000, 0.003)
problem = IsingProblem(h, J)
schedule = linear_schedule(1000)
workspace = Workspace(problem)

solve!(workspace, problem, schedule)    # the first call compiles solve!
println("allocated (bytes): ", @allocated solve!(workspace, problem, schedule))
@printf("objective:         %.2f\n", objective(problem, workspace.σ))
```

`workspace.σ` holds the solution in the Ising form. Convert a QUBO once with
`IsingProblem(qubo)` before such a loop.

The same workspace serves a new problem of the same size. Building a problem
copies nothing, so a problem with new linear terms and the same `J` is
cheap:

```@example performance
for seed in 1:3
    new_h = randn(MersenneTwister(seed), 2000)
    new_problem = IsingProblem(new_h, J)
    solve!(workspace, new_problem, schedule)
    @printf("objective: %.2f\n", objective(new_problem, workspace.σ))
end
```

## Single steps

`step!` runs a single step, and `evolve!` runs a whole schedule from the
current state, so you can watch a run. A new workspace holds the starting
state, with every value at 0. Every 200 steps, the following loop prints the
share of values that are already beyond ±0.9, and the objective at the
values rounded to their signs:

```@example performance
workspace = Workspace(problem)
for k in 1:length(schedule)
    step!(workspace, problem, schedule.β[k], schedule.γ[k])
    if k % 200 == 0
        values = workspace.state.z
        settled = count(v -> abs(v) > 0.9, values) / length(values)
        @printf("step %4d: %5.1f%% beyond ±0.9, objective %.2f\n",
                k, 100settled, objective(problem, sign.(values)))
    end
end
```
