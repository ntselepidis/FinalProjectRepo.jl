# Usage: `julia <script>.jl {gpu,cpu} {strong,weak}`
# e.g. `julia <script>.jl gpu strong
if !@isdefined USE_GPU
    const USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"
end

using DataFrames
using CSV
using ImplicitGlobalGrid
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
if !ParallelStencil.is_initialized()
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 3)
    else
        @init_parallel_stencil(Threads, Float64, 3)
    end
end
include("part1_kernel_programming.jl")

MPI.Init()

# Tolerance experiments reveal that 1e-6 gives sufficiently accurate results for this problem.
const SOLVER_TOL = 1e-6

for strong_scaling in [true, false]
    # We need 7 buffers for the solver
    # memory(nx) = nx^3 * sizeof(Float64) * 7
    # => with nx=ny=nz=2^9 we have memory(nx) ≈ 7.5GB => Fits on Titan X (12 GB)
    # After experimenting with 2^9 we found it very hard to converge.
    # Empirically, nx=ny=nz=2^7 gives a good trade-off between convergence speed and precision.
    nx, ny, nz = 
        if strong_scaling
            dims_dict = Dict(
                             1 => (1, 1, 1),
                             2 => (2, 1, 1),
                             4 => (2, 2, 1),
                             8 => (2, 2, 2),
                            )
            2^7 .÷ dims_dict[MPI.Comm_size(MPI.COMM_WORLD)]  # global, gets divided
        else
            2^7 .* (1, 1, 1)  # local, gets solved on every node
        end

    filename = joinpath("benchmark-results", "bench_diffusion_scaling_$(USE_GPU ? "gpu" : "cpu").csv")

    for use_shared_memory in [true, false]
        _, _, bench_res = diffusion_3D_kernel_programming(nx=nx, ny=ny, nz=nz, ttot=2.0, tol=SOLVER_TOL, use_shared_memory=use_shared_memory, init_and_finalize_MPI=false, verbose=true, scale_physical_size=(strong_scaling ? false : true))
        bench_row = (delta_t=bench_res.Δt,
                     Work=bench_res.Work,
                     Performance=bench_res.Performance,
                     Memory=bench_res.Memory,
                     Intensity=bench_res.Intensity,
                     Throughput=bench_res.Throughput,
                     use_shared_memory=use_shared_memory,
                     use_gpu=USE_GPU,
                     strong_scaling=strong_scaling, 
                     n_threads=Threads.nthreads(), 
                     n_mpi_ranks=MPI.Comm_size(MPI.COMM_WORLD))

        # Note that this is potentially susceptible to a race condition
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            bench_df = 
                if isfile(filename)
                    DataFrame(CSV.File(filename))
                else
                    DataFrame()
                end
            push!(bench_df, bench_row)

            open(filename, "w") do io
                CSV.write(io, bench_df)
            end
        end
    end
end

MPI.Finalize()
