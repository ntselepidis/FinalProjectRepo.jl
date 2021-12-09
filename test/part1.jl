# Testing part 1
using Test
using ReferenceTests
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3)

USE_GPU=false
include("../scripts-part1/part1_kernel_programming.jl") # modify to include the correct script
include("../scripts-part1/part1_array_programming.jl") # modify to include the correct script

# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])

MPI.Init()
Xc_g, H_g = diffusion_3D_array_programming(nx=32, ny=32, nz=32, init_and_finalize_MPI=false)
inds = Int.(ceil.(LinRange(1, length(Xc_g), 12)))
d_kernel = Dict(:X=> Xc_g[inds], :H_g=>H_g[inds, inds, 15])

Xc_g, H_g = diffusion_3D_array_programming(nx=32, ny=32, nz=32, init_and_finalize_MPI=false)
inds = Int.(ceil.(LinRange(1, length(Xc_g), 12)))
d_array = Dict(:X=> Xc_g[inds], :H_g=>H_g[inds, inds, 15])
MPI.Finalize()

@testset "Ref-file" begin
    @test_reference "reftest-files/test_1.bson" d_array by=comp
    @test_reference "reftest-files/test_1.bson" d_kernel by=comp
end
