if !@isdefined USE_GPU
    const USE_GPU = true
end

using DataFrames
using CSV
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
if !ParallelStencil.is_initialized()
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
    else
        @init_parallel_stencil(Threads, Float64, 2)
    end
end
include("part2.jl")

const FILENAME = joinpath("benchmark-results", "part2_semi_implicit_vs_explicit_experiment_results.csv")

const NX, NY = begin
    local nx, ny
    nx = 2^11
    ny = (nx ÷ 4)
    (nx, ny) .+ 1
end
@assert NX == (2^11 + 1)
@assert NY == (2^9 + 1)

# Pr_{liquid metals} ≈ 0.004 to 0.03
# Pr_{air} ≈ 0.7
# Pr_{water} ≈ 1.7 to 12
# Pr_{rock} ≈ practically infinite, i.e. 10^24
for Pr in 10 .^(-3.0:1.0:1.0)
    for beta in [0.0, 0.5, 1.0]
        println("Running for Pr=$(Pr) and beta=$(beta).")
        opt = SimIn_t()
        opt.nx = NX
        opt.ny = NY
        opt.Pr = Pr
        opt.tol = 1.0e-7
        opt.beta = beta
        opt.ttot = 0.005

        res = navier_stokes_2D(opt=opt)

        bench_row = (nx=NX, ny=NY, Pr=Pr, beta=beta, t_elapsed=res.t_elapsed, timed_iters=res.timed_iters)

        bench_df = 
            if isfile(FILENAME)
                DataFrame(CSV.File(FILENAME))
            else
                DataFrame()
            end
        push!(bench_df, bench_row)

        open(FILENAME, "w") do io
            CSV.write(io, bench_df)
        end
    end
end


