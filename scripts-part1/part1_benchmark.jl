@assert !@isdefined USE_GPU
const USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"

using CUDA
import MPI
using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using BenchmarkTools
using DataFrames
using CSV
using Statistics
using Pidfile

algorithm = ""
try
    global algorithm
    algorithm = ARGS[2]
catch
    println("Usage: julia --project scripts/part1_benchmark.jl [cpu/gpu] [array/kernel]")
    exit()
end

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

BenchTask = @NamedTuple {nx::Int, ny::Int, nz::Int}
BenchResult = @NamedTuple {
    device::String,
    algorithm::String,
    MPI_world_size::Int,
    n_threads::Int,
    nx::Int,
    ny::Int,
    nz::Int,
    mean::Float64,
    median::Float64,
    min::Float64,
    max::Float64,
    std::Float64,
}
tasks = DataFrame([BenchTask(round.(Int, (2^e, 2^e, 2^e))) for e = 4:0.5:7])

MPI.Init()
for t in eachrow(tasks)
    nx_, ny_, nz_ = t

    b = begin
        if algorithm == "array"
            @benchmark diffusion_3D_array_programming(nx = $nx_, ny = $ny_, nz = $nz_, verbose= false, init_and_finalize_MPI = false)
        elseif algorithm == "kernel"
            @benchmark diffusion_3D_kernel_programming(nx = $nx_, ny = $ny_, nz = $nz_, verbose= false, init_and_finalize_MPI = false)
        end
    end

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        bench_res = BenchResult((
            device = (USE_GPU ? "gpu" : "cpu"),
            algorithm = algorithm,
            MPI_world_size = MPI.Comm_size(MPI.COMM_WORLD),
            n_threads = Threads.nthreads(),
            nx = nx_,
            ny = ny_,
            nz = nz_,
            mean = mean(b).time * 1e-9,
            median = median(b).time * 1e-9,
            min = minimum(b).time * 1e-9,
            max = maximum(b).time * 1e-9,
            std = std(b).time * 1e-9,
        ))

        mkpidlock("bench_results.lck") do
            results_df = begin
                try
                    DataFrame(CSV.File("bench_results.csv"))
                catch
                    DataFrame()
                end
            end

            results_df = vcat(results_df, DataFrame([bench_res]))
            CSV.write("bench_results.csv", results_df)
        end
        println(results_df)
    end
end
MPI.Finalize()
