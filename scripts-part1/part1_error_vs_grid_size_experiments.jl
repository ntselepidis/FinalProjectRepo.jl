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

const LOCATION_OF_INTEREST = (4.5, 4.5, 4.5)

const FILENAME = joinpath("benchmark-results", "error_vs_grid_size_experiment_results.csv")

# Tolerance experiments reveal that 1e-6 gives sufficiently accurate results for this problem.
const SOLVER_TOL = 1e-6

MPI.Init()
for e in 4.0:0.5:7.0
    nx = ny = nz = Int(round(2^e))
    X, H, _ = diffusion_3D_kernel_programming(nx=nx, ny=ny, nz=nz, ttot=2.0, tol=SOLVER_TOL, init_and_finalize_MPI=false, verbose=false, scale_physical_size=false)
    
    dx = X[2] - X[1]

    interp_value = linear_interpolate_3D(H, dx)

    ix, iy, iz = round.(Int, (LOCATION_OF_INTEREST./ dx) .+ 1)

    df_row = (nx=nx, val=H[ix, iy, iz], interp_val=interp_value)

    # Note that this is potentially susceptible to a race condition
    res_df = 
        if isfile(FILENAME)
            DataFrame(CSV.File(FILENAME))
        else
            DataFrame()
        end
    push!(res_df, df_row)

    open(FILENAME, "w") do io
        CSV.write(io, res_df)
    end
end

MPI.Finalize()
