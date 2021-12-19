if !@isdefined USE_GPU
    const USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"
end
using ReferenceTests, Test
using CUDA
using LinearAlgebra
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
if !ParallelStencil.is_initialized()
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
    else
        @init_parallel_stencil(Threads, Float64, 2)
    end
end
include("../scripts-part2/multigrid.jl")

@testset "Test backslash" begin
    nx = 6
    ny = 6
    h = 1 / (ny - 1)
    x = rand((nx-2)*(ny-2))
    A = stencil_5pt(nx-2, ny-2) / h^2
    b = A * x
    xhat = A \ b
    @show norm(x - xhat) / norm(x)
    @test norm(x - xhat) / norm(x) < 1e-10
end

@testset "Test Multigrid with policy=$(execution_policy) and coarse_solve_size = $((2^l)+1) and nx=ny=$((2^k)+1) on $(USE_GPU ? "GPU" : "CPU")." for execution_policy in [parallel, parallel_shmem], l in 2:3, k in 7:10
    n = (2^k)+1
    h = 1 / (n - 1)
    c = 0.0
    tol = 1e-6

    opt = MGOpt()
    opt.execution_policy = execution_policy
    opt.coarse_solve_size = (2^l)+1

    inn = CartesianIndices((2:n-1, 2:n-1))

    xref = zeros(n, n)
    xref[inn] .= rand(n-2, n-2)
    b = zeros(n, n)
    b_ = zeros(n-2, n-2)
    A = stencil_5pt(n-2, n-2) / h^2
    b_[:] .= A * xref[inn][:]
    b[inn] .= b_

    xhat = @zeros(n, n)
    r_rms = MGsolve_2DPoisson!(xhat, Data.Array(b), h, c, tol, 20, false; opt=opt, verbose=true)
    @synchronize()
    xhat = Array(xhat)

    @test r_rms < tol * sqrt( sum(b.^2) / (n*n) )
    #@test norm(xref - xhat) / norm(xref) < tol
end

@testset "Test Jacobi solver" for execution_policy in [parallel, parallel_shmem]
    n = 33
    h = 1 / (n - 1)
    c = 0.0
    alpha = 1.0
    tol = 1e-6
    Nmax = 10000

    inn = CartesianIndices((2:n-1, 2:n-1))

    # CPU
    xref = rand(n, n)
    xref[1, :] .= 0.
    xref[n, :] .= 0.
    xref[:, 1] .= 0.
    xref[:, n] .= 0.

    b = zeros(n, n)
    b_ = zeros(n-2, n-2)

    A = stencil_5pt(n-2, n-2) / h^2
    b_[:] .= A * xref[inn][:]
    b[inn] .= b_

    # XPU
    b = Data.Array(b)
    xhat = @zeros(n, n)
    tolb = tol * sqrt(sum(b.^2)/(n*n))
    for i = 1 : Nmax
        res_rms = iteration_2DPoisson!(xhat, b, h, c, execution_policy)
        @synchronize
        if res_rms < tolb
            println("iter = $(i)")
            break
        end
    end
    xhat = Array(xhat)

    @test norm(xref - xhat) / norm(xref) < tolb
end

@testset "Test residual_2DPoisson" for execution_policy in [parallel, parallel_shmem]
    # CPU matrix
    # XPU residual
    n = 64
    h = 1 / (n - 1)
    c = 3.1415
    inn = CartesianIndices((2:n-1, 2:n-1))

    # XPU
    u = @rand(n, n)
    u[1, :] .= 0.
    u[n, :] .= 0.
    u[:, 1] .= 0.
    u[:, n] .= 0.
    f = @rand(n, n)
    res = @zeros(n, n)

    residual_2DPoisson_wrapper!(u, f, h, c, res, execution_policy)
    @synchronize()

    res_cpu = Array(res)

    # CPU

    u_cpu = Array(u)
    u_ = u_cpu[inn]

    f_cpu = Array(f)
    f_ = f_cpu[inn]

    A = stencil_5pt(n-2, n-2) / h^2 - c * I

    res_ = zeros(n-2, n-2)
    res_[:] .= A * u_[:] - f_[:]

    @test res_ ≈ res_cpu[inn]
end

@ParallelStencil.reset_parallel_stencil()
