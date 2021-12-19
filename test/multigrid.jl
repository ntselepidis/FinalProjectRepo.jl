using ReferenceTests, Test
using CUDA
using LinearAlgebra
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
if !ParallelStencil.is_initialized()
    @init_parallel_stencil(Threads, Float64, 2)
end
include("../scripts-part2/multigrid.jl")

@testset "Test backslash" begin
    nx = 6
    ny = 6
    h = 1 / (ny - 1)
    # x = Array(1.0 : (nx-2)*(ny-2))
    x = rand((nx-2)*(ny-2))
    A = stencil_5pt(nx-2, ny-2) / h^2
    b = A * x
    xhat = A \ b
    @show norm(x - xhat) / norm(x)
    @test norm(x - xhat) / norm(x) < 1e-10
end

@testset "Test Multigrid" begin
    k = 12
    n = (2^k)+1
    h = 1 / (n - 1)
    c = 0.0
    tol = 1e-6

    inn = CartesianIndices((2:n-1, 2:n-1))

    #xref = @zeros(n, n)
    #xref[inn] .= rand(n-2, n-2)
    xref = rand(n,n)
    b = zeros(n, n)
    b_ = b[inn]
    # xref = reshape(Data.Array(1.0 : (n-2)^2), n-2, n-2)
    A = stencil_5pt(n-2, n-2) / h^2
    b_[:] .= A * xref[inn][:]
    b[inn] .= b_

    xhat = @zeros(n, n)
    r_rms = MGsolve_2DPoisson!(xhat, Data.Array(b), h, c, tol, 200, false)
    #@test r_rms < tol
    #@test norm(xref - xhat) / norm(xref) < tol
end

@testset "Test Jacobi solver" begin
    n = 33
    h = 1 / (n - 1)
    c = 0.0
    alpha = 1.0
    tol = 1e-6
    Nmax = 10000

    inn = CartesianIndices((2:n-1, 2:n-1))

    xref = @zeros(n, n)
    xref[inn] .= rand(n-2, n-2)
    b = @zeros(n, n)
    b_ = b[inn]
    # xref = reshape(Data.Array(1.0 : (n-2)^2), n-2, n-2)
    A = stencil_5pt(n-2, n-2) / h^2
    b_[:] .= A * xref[inn][:]
    b[inn] .= b_

    xhat = @zeros(n, n)
    tolb = tol * sqrt(sum(b.^2)/(n*n))
    for i = 1 : Nmax
        res_rms = iteration_2DPoisson!(xhat, b, h, c, alpha)
        if res_rms < tolb
            println("iter = $(i)")
            break
        end
    end
    @test norm(xref - xhat) / norm(xref) < tolb
end

@testset "Test residual_2DPoisson" begin
    n = 64
    h = 1 / (n - 1)
    c = 3.1415
    inn = CartesianIndices((2:n-1, 2:n-1))

    u = @zeros(n, n)
    u[inn] .= rand(n-2, n-2)

    f = @rand(n, n)

    res = @zeros(n, n)

    @parallel residual_2DPoisson!(u, f, h, c, res)

    A = stencil_5pt(n-2, n-2) / h^2 - c * I

    res_ = @zeros(n-2, n-2)
    u_ = u[inn]
    res_[:] .= A * u_[:] - f[inn][:]

    @test res_ ≈ res[inn]
end

@ParallelStencil.reset_parallel_stencil()
