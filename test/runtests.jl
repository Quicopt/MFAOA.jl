using Test
using Random
using MFAOA
using MFAOA: SpinState, readout!, nvariables

# A plain reference implementation with explicit 3×3 rotation matrices, in the
# conventions documented at the top of src/kernel.jl.
function reference_evolve(h, J, Δ, β, γ)
    N = length(h)
    n = [[1.0, 0.0, 0.0] for _ in 1:N]
    for k in eachindex(β)
        z = [n[i][3] for i in 1:N]
        m = h + J * z
        for i in 1:N
            θ = 2 * γ[k] * m[i]
            Rz = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]
            φ = 2 * β[k] * Δ[i]
            Rx = [1 0 0; 0 cos(φ) sin(φ); 0 -sin(φ) cos(φ)]
            n[i] = Rx * (Rz * n[i])
        end
    end
    n
end

# Exhaustive optimum of a small problem, in the problem's own sense.
function brute_force(prob::IsingProblem)
    n = nvariables(prob)
    better = prob.sense === :max ? (>) : (<)
    best = prob.sense === :max ? -Inf : Inf
    bestσ = Int8[]
    σ = Vector{Int8}(undef, n)
    for bits in 0:(2^n - 1)
        for i in 1:n
            σ[i] = (bits >> (i - 1)) & 1 == 1 ? 1 : -1
        end
        E = objective(prob, σ)
        if better(E, best)
            best = E
            bestσ = copy(σ)
        end
    end
    best, bestσ
end

function random_ising(rng, N; Δ = nothing)
    J = randn(rng, N, N)
    J = (J + J') / 2
    for i in 1:N
        J[i, i] = 0
    end
    IsingProblem(randn(rng, N), J; Δ = Δ)
end

alloc_solve!(ws, prob, sched) = @allocated solve!(ws, prob, sched)

@testset "MFAOA" begin

@testset "schedule" begin
    s = linear_schedule(4, 0.5)
    @test s.γ ≈ 0.5 .* [1, 2, 3, 4] ./ 4
    @test s.β ≈ 0.5 .* (1 .- [0, 1, 2, 3] ./ 4)
    @test length(s) == 4
    @test eltype(linear_schedule(3; T = Float32).β) == Float32
    @test eltype(Schedule([1, 2], [3, 4]).β) == Float64
    @test_throws ArgumentError Schedule([0.1, 0.2], [0.1])
    @test_throws ArgumentError Schedule(Float64[], Float64[])
    @test_throws ArgumentError linear_schedule(0)
end

@testset "IsingProblem validation and objective" begin
    J = [0.0 1.0; 1.0 0.0]
    @test_throws ArgumentError IsingProblem([1.0], J)                          # size mismatch
    @test_throws ArgumentError IsingProblem([1.0, 2.0], [0.0 1.0; 2.0 0.0])    # not symmetric
    @test_throws ArgumentError IsingProblem([1.0, 2.0], [1.0 1.0; 1.0 0.0])    # diagonal
    @test_throws ArgumentError IsingProblem([1.0, 2.0], J; Δ = [1.0])         # Δ length
    @test_throws ArgumentError IsingProblem([1.0, 2.0], J; Δ = [1.0, 0.0])    # Δ not positive
    @test_throws ArgumentError IsingProblem([1.0, 2.0], J; sense = :maximize)
    @test_throws ArgumentError IsingProblem(Float64[], zeros(0, 0))

    prob = IsingProblem([1, -2], [0 3; 3 0])                                   # integers promote
    @test eltype(prob) == Float64
    @test objective(prob, [1, 1]) == 1 - 2 + 3
    @test objective(prob, [1, -1]) == 1 + 2 - 3

    pmax = IsingProblem([1, -2], [0 3; 3 0]; offset = 10, sense = :max)
    @test objective(pmax, [1, 1]) == 1 - 2 + 3 + 10                           # reported as stated
    @test pmax.h == [-1.0, 2.0]                                                # stored for minimization
    @test pmax.offset == -10.0
end

@testset "QUBO and Ising forms agree on every x" begin
    rng = MersenneTwister(7)
    n = 5
    Q = randn(rng, n, n)                                   # asymmetric, nonzero diagonal
    c = randn(rng, n)
    for sense in (:min, :max)
        q = QUBOProblem(Q; c = c, constant = 0.3, sense = sense)
        ising = IsingProblem(q)
        @test ising.sense === sense
        for bits in 0:(2^n - 1)
            x = [(bits >> (i - 1)) & 1 for i in 1:n]
            σ = 2 .* x .- 1
            @test objective(ising, σ) ≈ objective(q, x) atol = 1e-12
        end
    end
    @test objective(QUBOProblem([2.0 0.0; 0.0 0.0]; c = [1.0, 0.0], constant = 0.5), [1, 0]) == 3.5
    @test_throws ArgumentError QUBOProblem(Q; c = [1.0])
    @test_throws ArgumentError QUBOProblem(Q[:, 1:2])
    @test_throws ArgumentError QUBOProblem(Q; sense = :maximise)
end

@testset "kernel matches the reference rotations" begin
    rng = MersenneTwister(11)
    N = 6
    for Δ in (nothing, 0.5 .+ rand(rng, N))
        prob = random_ising(rng, N; Δ = Δ)
        sched = Schedule(rand(rng, 20), rand(rng, 20))      # two independent, arbitrary schedules
        res = solve(prob, sched)
        ref = reference_evolve(prob.h, prob.J, Δ === nothing ? ones(N) : Δ, sched.β, sched.γ)
        for i in 1:N
            n_i = [res.state.x[i], res.state.y[i], res.state.z[i]]
            @test n_i ≈ ref[i] atol = 1e-12
            @test sqrt(sum(abs2, n_i)) ≈ 1 atol = 1e-12
        end
        @test res.σ == Int8.(sign.(res.state.z))
        @test res.objective == objective(prob, res.σ)
    end
end

@testset "sign convention: a positive field pushes the spin to -1" begin
    @test solve(IsingProblem([1.0], zeros(1, 1)); p = 50).σ == [-1]
    @test solve(IsingProblem([-1.0], zeros(1, 1)); p = 50).σ == [1]
    @test solve(IsingProblem([1.0], zeros(1, 1); sense = :max); p = 50).σ == [1]
end

@testset "readout breaks ties by the field" begin
    s = SpinState{Float64}([0.0, 0.0, 0.0, 0.6], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, -0.8])
    σ = zeros(Int8, 4)
    readout!(σ, s, [1.0, -1.0, 0.0, 5.0])
    @test σ == [-1, 1, 1, -1]
end

@testset "agrees with QAOA.jl's mean-field evolution" begin
    # The 5-spin instance of QAOA.jl's test suite with spin 5 fixed to +1, so
    # that h = J[1:4, 5]. QAOA.jl maximizes h'σ + σ'Jσ/2, which in our form is
    # the minimization of (-h, -J); its y components then come out with the
    # opposite sign and everything else is identical.
    J5 = [0.0 -0.12793573348166293 -0.21670411503453044 -1.1866001328940943 -0.0037049988892551987;
          -0.12793573348166293 0.0 0.14105236137693697 0.1882996047044451 -0.4765521402619942;
          -0.21670411503453044 0.14105236137693697 0.0 0.25097277081028896 -0.5838590431585854;
          -1.1866001328940943 0.1882996047044451 0.25097277081028896 0.0 -0.3056466236010491;
          -0.0037049988892551987 -0.4765521402619942 -0.5838590431585854 -0.3056466236010491 0.0]
    h = -J5[1:4, 5]
    J = -J5[1:4, 1:4]
    p = 10
    τ = 0.5
    γ = collect(τ .* ((1:p) .- 1 / 2) ./ p)
    β = collect(τ .* (1 .- (1:p) ./ p))
    β[p] = τ / (4p)
    res = solve(IsingProblem(h, J), Schedule(β, γ))
    S_test = [[-0.4280189887648497, -0.57845514021309, 0.6943985858408479],
              [-0.4161436735954462, -0.1965348184395327, -0.8878054449413042],
              [-0.3151885316091266, 0.14809317283731896, -0.9374031159010826],
              [0.06825210700086091, 0.5908423629410464, -0.8038948638001017]]
    for i in 1:4
        @test [res.state.x[i], -res.state.y[i], res.state.z[i]] ≈ S_test[i] rtol = 1e-10
    end
    @test res.objective_mf ≈ -2.5367551470078142 rtol = 1e-10
    @test res.σ == [1, -1, -1, -1]
end

@testset "finds the optimum of easy instances" begin
    N = 8
    chain = zeros(N, N)
    for i in 1:(N - 1)
        chain[i, i + 1] = chain[i + 1, i] = 1
    end
    # Ferromagnetic chain (negative couplings favour alignment) in a weak field.
    ferro = IsingProblem(fill(-0.1, N), -chain)
    res = solve(ferro; p = 500)
    @test res.σ == fill(1, N)
    @test res.objective == brute_force(ferro)[1]
    @test res.objective_mf ≈ res.objective atol = 1e-3   # spins near, not on, the poles
    # Antiferromagnetic chain with a field on the first spin only.
    h = zeros(N); h[1] = -0.5
    anti = IsingProblem(h, chain)
    res = solve(anti; p = 1000)
    @test res.σ == [isodd(i) ? 1 : -1 for i in 1:N]
    @test res.objective == brute_force(anti)[1]
    # Same instances as maximization problems of the negated objective.
    ferro_max = IsingProblem(fill(0.1, N), chain; sense = :max)
    @test solve(ferro_max; p = 500).σ == fill(1, N)
    @test solve(ferro_max; p = 500).objective == brute_force(ferro_max)[1]
end

@testset "flip-symmetric problems are rejected" begin
    J = [0.0 -1.0; -1.0 0.0]
    prob = IsingProblem([0.0, 0.0], J)
    @test isflipsymmetric(prob)
    @test_throws ArgumentError solve(prob)
    @test !isflipsymmetric(IsingProblem([1e-3, 0.0], J))
    @test isflipsymmetric(IsingProblem([1e-20, 0.0], J))     # zero up to rounding
    @test isflipsymmetric(IsingProblem([0.0], zeros(1, 1)))  # constant objective
    # Max-cut of a triangle: cut(x) = Σ_(i,j) (x_i + x_j - 2 x_i x_j) is
    # invariant under x → 1 - x, so all fields vanish after conversion.
    Q = [0 -2 -2; 0 0 -2; 0 0 0]
    cut = QUBOProblem(Q; c = [2, 2, 2], sense = :max)
    @test IsingProblem(cut).h == zeros(3)
    @test isflipsymmetric(IsingProblem(cut))
    @test_throws ArgumentError solve(cut)
end

@testset "solve! allocates nothing" begin
    rng = MersenneTwister(3)
    N = 32
    for Δ in (nothing, 1 .+ rand(rng, N))
        prob = random_ising(rng, N; Δ = Δ)
        sched = linear_schedule(50, 0.5)
        ws = Workspace(prob)
        solve!(ws, prob, sched)                 # compile
        @test alloc_solve!(ws, prob, sched) == 0
        @test_throws ArgumentError solve!(Workspace(random_ising(rng, N + 1)), prob, sched)
    end
end

@testset "solve defaults and Result" begin
    prob = IsingProblem([-1.0, 0.5], [0.0 -0.3; -0.3 0.0])
    res = solve(prob)                            # p = 1000, τ = 0.5
    @test res.x == (res.σ .+ 1) .÷ 2
    @test res.objective == objective(prob, res.σ)
    @test res.objective == brute_force(prob)[1]
    @test res.objective_mf ≈ res.objective atol = 1e-3
    @test occursin("objective", sprint(show, res))
    @test occursin("2 variables", sprint(show, prob))

    q = QUBOProblem([1.0 -2.0; 0.0 1.0]; c = [-1.0, -1.0])
    r = solve(q; p = 200)
    @test r.objective ≈ objective(q, r.x)
    @test r.objective == brute_force(IsingProblem(q))[1]
    @test occursin("2 variables", sprint(show, q))
    @test solve(q, linear_schedule(200)).σ == r.σ
end

end # MFAOA
