# Part 1 of final project: Diffusion equation
if !@isdefined USE_GPU
    const USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"
end
using CUDA
import MPI
using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

## Trick so we can include this file with cpu or gpu
if abspath(PROGRAM_FILE) == @__FILE__
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 3)
        println("Init parallel stencil for GPU.")
    else
        @init_parallel_stencil(Threads, Float64, 3)
        println("Init parallel stencil for CPU.")
    end
end

include("part1_array_programming.jl")
include("part1_kernel_programming.jl")

function main()
    local nx_, ny_, nz_ = 32, 32, 32
    local algorithm = ""
    local bench = "0"
    try
        algorithm = ARGS[2]
    catch
        println("Usage: mpiexecjl -np <np> julia --project scripts/part1.jl [cpu/gpu] [array/kernel] [nx ny nz]")
        exit()
    end

    if length(ARGS) > 2
        try
            nx_, ny_, nz_ = parse.(Int, ARGS[3:5])
        catch
            println("Usage: mpiexecjl -np <np> julia --project scripts/part1.jl [cpu/gpu] [array/kernel] [nx ny nz]")
            exit()
        end
    end
    if length(ARGS) > 5
        bench = ARGS[6]
    end
    bench = (bench == "1")

    if algorithm == "array"
        _, _, bench_res = diffusion_3D_array_programming(nx = nx_, ny = ny_, nz = nz_, verbose=!bench)
    elseif algorithm == "kernel"
        _, _, bench_res = diffusion_3D_kernel_programming(nx = nx_, ny = ny_, nz = nz_, verbose=!bench)
    else
        println("Usage: mpiexecjl -np <np> julia --project scripts/part1.jl [cpu/gpu] [array/kernel] [nx ny nz]")
        exit()
    end
    if bench
        @show bench_res
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
