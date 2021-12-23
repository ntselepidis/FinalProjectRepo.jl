if !@isdefined USE_GPU
    const USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"
end
using CUDA
using LinearAlgebra
using DataFrames, CSV
using BenchmarkTools, Statistics
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


const FILENAME = "bench_multigrid_$(USE_GPU ? "gpu" : "cpu").csv"
if isfile(FILENAME)
    bench_df = DataFrame(CSV.File(FILENAME))
else
    bench_df = DataFrame()
end
#for execution_policy in [parallel, parallel_shmem], l in 2:3, k in 7:10
for k in 4:13, l in 2:min(k-l, 8), solver in [jacobi, conjugate_gradient], execution_policy in [parallel, parallel_shmem]
    @info "Running with params" USE_GPU, execution_policy, l, k, solver
    n = (2^k)+1
    h = 1 / (n - 1)
    c = 0.0
    tol = 1e-6
    Nmax = 100

    opt = MGOpt()
    opt.execution_policy = execution_policy
    opt.coarse_solve_size = (2^l)+1
    opt.coarse_solver = solver

    for seed in 1:1
        try
            bench_res = @benchmark MGsolve_2DPoisson!(x, b, $h, $c, $tol, $Nmax, false; opt=$opt, verbose=false) setup=((x, b) = (@zeros($n, $n), @rand($n, $n))) evals=1
            bench_row = (
                 execution_policy = string(execution_policy),
                 coarse_solver = string(solver),
                 k = k,  # exponent for computing size of largest grid with nx = (2^k)+1 
                 l = l,  # same as k but for the coarsest grid
                 median_time = median(bench_res).time / 1e9,
                 mean_time = mean(bench_res).time / 1e9,
                 std_time = std(bench_res).time / 1e9,
                 seed = seed,
                 use_gpu = USE_GPU,
                 nthreads = Threads.nthreads(),
                )
            push!(bench_df, bench_row)

            open(FILENAME, "w") do io
                CSV.write(io, bench_df)
            end
        catch e
            @warn "Failed with " e
        end 
    end
end
