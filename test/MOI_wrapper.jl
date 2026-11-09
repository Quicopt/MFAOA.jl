import MathOptInterface as MOI
using JuMP
using LinearAlgebra: triu

@testset "JuMP: min and max of a QUBO" begin
    model = Model(MFAOA.Optimizer)
    @variable(model, x[1:3], Bin)
    @objective(model, Min, 2x[1] * x[2] - 3x[2] * x[3] - x[1] - x[2] + x[3])
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test primal_status(model) == FEASIBLE_POINT
    @test result_count(model) == 1
    @test value.(x) == [0, 1, 1]
    @test objective_value(model) == -3
    @test solve_time(model) ≥ 0
    @test occursin("steps", raw_status(model))

    @objective(model, Max, -(2x[1] * x[2] - 3x[2] * x[3] - x[1] - x[2] + x[3]) + 1)
    optimize!(model)
    @test value.(x) == [0, 1, 1]
    @test objective_value(model) == 4
end

@testset "JuMP: objective forms" begin
    model = Model(MFAOA.Optimizer)
    set_silent(model)
    @variable(model, x[1:2], Bin)
    @objective(model, Min, x[1] - x[2])               # affine
    optimize!(model)
    @test value.(x) == [0, 1]
    @test objective_value(model) == -1
    @objective(model, Max, x[1])                      # a single variable
    optimize!(model)
    @test value(x[1]) == 1
    @test objective_value(model) == 1
    @objective(model, Min, x[1]^2 - 2x[1] + 0.5)      # diagonal term: x² = x on binaries
    optimize!(model)
    @test value(x[1]) == 1
    @test objective_value(model) == -0.5
end

@testset "JuMP: fixing variables breaks the symmetry" begin
    # Max-cut of a triangle: the cut value is invariant under swapping sides.
    edges = [(1, 2), (2, 3), (1, 3)]
    model = Model(MFAOA.Optimizer)
    @variable(model, x[1:3], Bin)
    @objective(model, Max, sum(x[i] + x[j] - 2x[i] * x[j] for (i, j) in edges))
    optimize!(model)
    @test termination_status(model) == INVALID_MODEL
    @test result_count(model) == 0
    @test occursin("every variable is flipped (x → 1 - x)", raw_status(model))
    @test occursin("with JuMP's `fix`, for example x[1].", raw_status(model))
    @test primal_status(model) == NO_SOLUTION

    fix(x[1], 1)
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test value(x[1]) == 1
    @test objective_value(model) == 2                 # every cut of a triangle has size 2
    @test objective_value(model) == sum(value(x[i]) + value(x[j]) - 2value(x[i]) * value(x[j]) for (i, j) in edges)

    fix(x[2], 0)
    fix(x[3], 0)
    optimize!(model)                                   # nothing left to solve
    @test termination_status(model) == OPTIMAL
    @test value.(x) == [1, 0, 0]
    @test objective_value(model) == 2
end

@testset "JuMP: every separate part needs a fixed variable" begin
    # Max-cut of two separate edges: fixing x[1] decides the first edge only.
    edges = [(1, 2), (3, 4)]
    model = Model(MFAOA.Optimizer)
    @variable(model, x[1:4], Bin)
    @objective(model, Max, sum(x[i] + x[j] - 2x[i] * x[j] for (i, j) in edges))
    fix(x[1], 0)
    optimize!(model)
    @test termination_status(model) == INVALID_MODEL
    @test occursin("the variables x[3] and x[4] are all flipped", raw_status(model))
    @test occursin("Fix one of them to 0 or 1 with JuMP's `fix`, for example x[3].", raw_status(model))
    fix(x[3], 1)
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test objective_value(model) == 2

    # Anonymous variables are named by their position in the model.
    model = Model(MFAOA.Optimizer)
    y = @variable(model, [1:2], Bin)
    @objective(model, Max, y[1] + y[2] - 2y[1] * y[2])
    optimize!(model)
    @test occursin("for example variable 1.", raw_status(model))
end

@testset "JuMP: bounds and integer variables" begin
    objective(x) = 2x[1] * x[2] - 3x[2] * x[3] - x[1] - x[2] + x[3]
    # Bounds that allow 0 and 1 change nothing.
    model = Model(MFAOA.Optimizer)
    @variable(model, 0 <= x[1:3] <= 1, Bin)
    @objective(model, Min, objective(x))
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test value.(x) == [0, 1, 1]
    # An integer variable with bounds 0 and 1 is binary.
    model = Model(MFAOA.Optimizer)
    @variable(model, 0 <= y[1:3] <= 1, Int)
    @objective(model, Min, objective(y))
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test value.(y) == [0, 1, 1]
    @test (VariableRef, MOI.Integer) in list_of_constraint_types(model)
    # A bound that allows only one of 0 and 1 holds the variable there.
    model = Model(MFAOA.Optimizer)
    @variable(model, x[1:3], Bin)
    set_lower_bound(x[1], 1)
    @objective(model, Min, objective(x))
    optimize!(model)
    @test value.(x) == [1, 1, 1]
    @test objective_value(model) == -2
    set_upper_bound(x[2], 0.5)
    set_upper_bound(x[3], 0)
    optimize!(model)
    @test termination_status(model) == OPTIMAL           # nothing left to solve
    @test value.(x) == [1, 0, 0]
    # An integer variable with more values than 0 and 1 is rejected.
    model = Model(MFAOA.Optimizer)
    @variable(model, 0 <= y <= 2, Int)
    @objective(model, Min, y)
    optimize!(model)
    @test termination_status(model) == INVALID_MODEL
    @test raw_status(model) == "y is neither binary nor fixed; MFAOA solves problems in binary variables only."
    # Bounds that allow neither 0 nor 1 leave no solution.
    model = Model(MFAOA.Optimizer)
    @variable(model, 0.2 <= z <= 0.8, Bin)
    @objective(model, Min, z)
    optimize!(model)
    @test termination_status(model) == INFEASIBLE
    @test raw_status(model) == "The bounds 0.2 ≤ z ≤ 0.8 allow neither 0 nor 1."
    @test result_count(model) == 0
end

@testset "JuMP: models outside the problem class" begin
    model = Model(MFAOA.Optimizer)
    @variable(model, x, Bin)
    @variable(model, y)                                # continuous
    @objective(model, Min, x * y - x)
    optimize!(model)
    @test termination_status(model) == INVALID_MODEL
    @test raw_status(model) == "y is neither binary nor fixed; MFAOA solves problems in binary variables only."

    model = Model(MFAOA.Optimizer)
    @variable(model, x[1:2], Bin)
    fix(x[1], 0.5)
    @objective(model, Min, x[1] * x[2])
    optimize!(model)
    @test termination_status(model) == INVALID_MODEL
    @test raw_status(model) == "x[1] is fixed to 0.5; MFAOA accepts variables fixed to 0 or 1 only."

    model = Model(MFAOA.Optimizer)
    @variable(model, x[1:2], Bin)
    # JuMP rejects the constraint as soon as it is added, with its own message.
    err = try @constraint(model, x[1] + x[2] <= 1); nothing catch e; e end
    @test err isa ErrorException && occursin("not supported", err.msg)

    model = Model(MFAOA.Optimizer)                     # no objective, no variables
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test objective_value(model) == 0
end

@testset "JuMP: units of the objective do not matter" begin
    # A ring of 12 variables with alternating couplings and small biases.
    function ring_model(weight)
        model = Model(MFAOA.Optimizer)
        @variable(model, x[1:12], Bin)
        @objective(model, Min, weight * (sum((-1)^i * x[i] * x[mod1(i + 1, 12)] for i in 1:12)
                                        - sum(0.1i * x[i] for i in 1:12)))
        optimize!(model)
        value.(x), objective_value(model)
    end
    x1, v1 = ring_model(1)
    x1000, v1000 = ring_model(1000)
    @test x1000 == x1
    @test v1000 ≈ 1000v1
end

@testset "JuMP: sparse and dense storage" begin
    # The wrapper keeps the couplings sparse below a fill of 25% and dense above.
    sparse_Q = sparse([1, 2], [2, 3], [1.0, -1.0], 10, 10)
    @test MFAOA._coupling_storage(sparse_Q) isa SparseMatrixCSC
    dense_Q = sparse(triu(ones(10, 10), 1))
    @test MFAOA._coupling_storage(dense_Q) isa Matrix

    # Fixing variables in a sparse model: substitution matches brute force.
    model = Model(MFAOA.Optimizer)
    @variable(model, x[1:8], Bin)
    @objective(model, Max, sum(x[i] + x[i + 1] - 2x[i] * x[i + 1] for i in 1:7) + 0.5x[8])
    fix(x[1], 1)
    fix(x[5], 1)
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    best = maximum(0:63) do bits
        y = [1; [(bits >> k) & 1 for k in 0:2]; 1; [(bits >> k) & 1 for k in 3:5]]
        sum(y[i] + y[i + 1] - 2y[i] * y[i + 1] for i in 1:7) + 0.5y[8]
    end
    @test objective_value(model) == best
    @test value(x[1]) == 1 && value(x[5]) == 1
end

@testset "JuMP: options" begin
    model = Model(optimizer_with_attributes(MFAOA.Optimizer, "p" => 200, "tau" => 0.4))
    @variable(model, x[1:2], Bin)
    @objective(model, Min, x[1] * x[2] - x[1])
    @test get_attribute(model, "p") == 200
    @test get_attribute(model, "tau") == 0.4
    optimize!(model)
    @test value.(x) == [1, 0]
    @test occursin("200 steps", raw_status(model))

    sched = linear_schedule(50, 0.5)
    set_attribute(model, "beta", sched.β)
    @test_throws ArgumentError optimize!(model)        # gamma missing
    set_attribute(model, "gamma", sched.γ)
    optimize!(model)
    @test occursin("50 steps", raw_status(model))
    @test get_attribute(model, "beta") == sched.β
    set_attribute(model, "beta", nothing)
    set_attribute(model, "gamma", nothing)
    optimize!(model)
    @test occursin("200 steps", raw_status(model))

    @test_throws ArgumentError set_attribute(model, "p", 0)
    @test_throws ArgumentError set_attribute(model, "tau", -1.0)
    @test_throws MOI.UnsupportedAttribute set_attribute(model, "layers", 3)
    @test sprint(show, MFAOA.Optimizer()) == "MFAOA.Optimizer"
    @test MOI.get(MFAOA.Optimizer(), MOI.SolverName()) == "MFAOA"
    @test MOI.get(MFAOA.Optimizer(), MOI.SolverVersion()) == string(pkgversion(MFAOA))
end

@testset "MOI.Test" begin
    optimizer = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MFAOA.Optimizer(),
    )
    model = MOI.Bridges.full_bridge_optimizer(optimizer, Float64)
    MOI.set(model, MOI.Silent(), true)
    config = MOI.Test.Config(;
        atol = 1e-6, rtol = 1e-6,
        # MFAOA returns a solution without a proof that it is optimal.
        optimal_status = MOI.LOCALLY_SOLVED,
        # MFAOA computes no duals, no basis and no bound on the optimal value.
        exclude = Any[MOI.ConstraintDual, MOI.DualObjectiveValue, MOI.ConstraintBasisStatus,
                      MOI.VariableBasisStatus, MOI.ObjectiveBound],
    )
    MOI.Test.runtests(model, config; exclude = [
        # MFAOA solves problems in binary variables only, and returns INVALID_MODEL
        # for these tests. They use continuous variables ...
        r"^test_linear_variable_open_intervals$",
        r"^test_modification_coef_scalar_objective$",
        r"^test_modification_const_scalar_objective$",
        r"^test_modification_delete_variable_with_single_variable_obj$",
        r"^test_modification_delete_variables_in_a_batch$",
        r"^test_modification_set_singlevariable_lessthan$",
        r"^test_modification_transform_singlevariable_lessthan$",
        r"^test_objective_FEASIBILITY_SENSE_clears_objective$",
        r"^test_objective_ObjectiveFunction_blank$",
        r"^test_objective_ObjectiveFunction_constant$",
        r"^test_objective_ObjectiveFunction_duplicate_terms$",
        r"^test_objective_ObjectiveFunction_VariableIndex$",
        r"^test_objective_qp_ObjectiveFunction_edge_cases$",
        r"^test_objective_qp_ObjectiveFunction_zero_ofdiag$",
        r"^test_solve_optimize_twice$",
        r"^test_solve_result_index$",
        r"^test_solve_TerminationStatus_DUAL_INFEASIBLE$",
        r"^test_variable_solve_with_lowerbound$",
        r"^test_variable_solve_with_upperbound$",
        # ... or integer variables with a single bound, which can take values
        # other than 0 and 1.
        r"^test_variable_solve_Integer_with_lower_bound$",
        r"^test_variable_solve_Integer_with_upper_bound$",
        # In these tests the bounds fix every variable. With nothing left to
        # search, MFAOA returns OPTIMAL instead of LOCALLY_SOLVED.
        r"^test_constraint_ZeroOne_bounds_2$",
        r"^test_variable_solve_ZeroOne_with_0_upper_bound$",
        r"^test_variable_solve_ZeroOne_with_1_lower_bound$",
        r"^test_variable_solve_ZeroOne_with_bounds_then_delete$",
    ])
end
