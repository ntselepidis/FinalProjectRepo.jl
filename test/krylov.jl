if !@isdefined USE_GPU
    const USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"
end

using Test
using LinearAlgebra

using ParallelStencil
if !ParallelStencil.is_initialized()
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
    else
        @init_parallel_stencil(Threads, Float64, 2)
    end
end

include("../scripts-part2/krylov.jl")

@testset "Test Krylov solver" for execution_policy in [parallel, parallel_shmem]
    @info "Runnint with " execution_policy
    n = 66
    h = 1 / (n - 1)
    c = 3.14
    tol = 1e-6
    Nmax = 1000

    b = @ones(n, n)
    b[1, :] .= 0.0
    b[n, :] .= 0.0
    b[:, 1] .= 0.0
    b[:, n] .= 0.0
    xhat = @zeros(n, n)
    res_rms = cg!(xhat, b, h, h, c, tol, Nmax, execution_policy=execution_policy)

    @test res_rms < tol * sqrt(sum(b.^2) / n^2)
end

@ParallelStencil.reset_parallel_stencil()
