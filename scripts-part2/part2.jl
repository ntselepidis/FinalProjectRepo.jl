# Part 2: another PDE
if !@isdefined USE_GPU
    const USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"
end
using JLD
using SparseArrays
using SuiteSparse
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

include("./part2_utils.jl")
include("./multigrid.jl")

""" Initial condition (for temperature T or vorticity W). """
@enum Init_t begin
    cosine       # Initalize field as cosine
    random       # Initialize field as random
    W_from_file  # Read initial condition from file (for W, only for debugging)
end

""" Input parameters for Navier-Stokes simulation. """
mutable struct SimIn_t
    k               :: Float64  # thermal diffusivity (always set k = 1.0 since we use the non-dimensionalized Navier-Stokes equations)
    Ra              :: Float64  # Rayleigh number
    Pr              :: Float64  # Prandtl number
    nx              :: Int      # number of grid-points in x-direction
    ny              :: Int      # number of grid-points in y-direction
    ttot            :: Float64  # total time for simulation
    beta            :: Float64  # explicit (beta = 0.0) / implicit (beta = 1.0) / semi-implicit (beta = 0.5)
    niters          :: Int      # max number of iterations for iterative solver
    tol             :: Float64  # tolerance for linear solver
    a_dif           :: Float64  # diffusive timestep parameter
    a_adv           :: Float64  # advective timestep parameter
    T_init_strategy :: Init_t   # initialization strategy for T
    W_init_strategy :: Init_t   # initialization strategy for W

    SimIn_t() = new(1.0, 1.0e6, 1.0e-3, 257, 65, 0.1, 0.0, 50, 1.0e-3, 0.15, 0.4, cosine, random)
end

""" Output fields of Navier-Stokes simulation. """
struct SimOut_t
    T :: Matrix{Float64}    # Temperature
    W :: Matrix{Float64}    # Vorticity
    S :: Matrix{Float64}    # Streamfunction
    t_elapsed :: Float64    # Computation time
    timed_iters :: Float64  # Number of timesteps timed (i.e. minus warmup)
end

""" Initializes array M (i.e. T or W) given initialization scheme, mesh size h, and width. """
@views function init_array!(M, scheme::Init_t, h, width)
    nx, ny = size(M)
    if scheme == cosine
        for i in 1:nx
            M[i, :] .= 0.5 * (1.0 + cos((3.0*π*(i-1)*h)/width))
        end
    elseif scheme == random
        M .= @rand(nx, ny)
    elseif scheme == W_from_file # only for testing against fortran
        M .= open(joinpath(@__DIR__, "..", "test", "reftest-files", "fortran", "Winit.bin")) do io
            return load(io)
        end
    else
        error()
    end
end

""" Computes dt for explicit or implicit timestepping scheme. """
@views function compute_dt(v, vx, vy, dt_dif, a_dif, a_adv, h, beta)
    v_max = maximum(v)
    if (v_max == 0)
        dt = dt_dif
    else
        # compute advective timestep
        dt_adv = a_adv * min(h / maximum(abs.(vx)), h / maximum(abs.(vy)))
        # compute timestep
        dt = (beta >= 0.5 ? dt_adv : min(dt_dif, dt_adv))
    end
    return dt
end

""" Computes velocity given streamfunction. """
@parallel_indices (ix, iy) function compute_velocity!(S, hx, hy, vx, vy)
    if (1 < ix < size(S, 1) && 1 < iy < size(S, 2))
        vx[ix, iy] =  ( S[ix, iy+1] - S[ix, iy-1] ) / ( 2 * hy )
        vy[ix, iy] = -( S[ix+1, iy] - S[ix-1, iy] ) / ( 2 * hx )
    end
    return nothing
end

""" Computes Ra * (dT / dx) term, used in vorticity update. """
@parallel_indices (ix, iy) function compute_Ra_dTdx!(Ra, hx, T, Ra_dTdx)
    if (1 < ix < size(T, 1) && 1 < iy < size(T, 2))
        Ra_dTdx[ix, iy] =  Ra * ( T[ix+1, iy] - T[ix-1, iy] ) / ( 2 * hx )
    end
    return nothing
end

""" Computes diffusion of 2D field. """
@parallel_indices (ix, iy) function compute_diffusion2d!(T, hx, hy, k, dT2)
    if (1 < ix < size(T, 1) && 1 < iy < size(T, 2))
        dT2[ix, iy] = k * (( T[ix+1, iy] - 2*T[ix, iy]+ T[ix-1, iy] ) / hx^2 +
                           ( T[ix, iy+1] - 2*T[ix, iy]+ T[ix, iy-1] ) / hy^2)
    end
    return nothing
end

""" Computes advection of 2D field in x-dimension (with upwinding). """
@parallel_indices (ix, iy) function compute_advection2d_x!(T, hx, vx, dTx)
    if (1 < ix < size(T, 1) && 1 < iy < size(T, 2))
        if vx[ix, iy] > 0
            dTx[ix, iy] = vx[ix, iy] * (T[ix, iy] - T[ix-1, iy]) / hx
        else
            dTx[ix, iy] = vx[ix, iy] * (T[ix+1, iy] - T[ix, iy]) / hx
        end
    end
    return nothing
end

""" Computes advection of 2D field in y-dimension (with upwinding). """
@parallel_indices (ix, iy) function compute_advection2d_y!(T, hy, vy, dTy)
    if (1 < ix < size(T, 1) && 1 < iy < size(T, 2))
        if vy[ix, iy] > 0
            dTy[ix, iy] = vy[ix, iy] * (T[ix, iy] - T[ix, iy-1]) / hy
        else
            dTy[ix, iy] = vy[ix, iy] * (T[ix, iy+1] - T[ix, iy]) / hy
        end
    end
    return nothing
end

""" Runs Navier-Stokes simulation in 2D. """
@views function navier_stokes_2D(; opt :: SimIn_t = SimIn_t(), verbose=true, do_vis=false, testmode=false)
    nx, ny = opt.nx, opt.ny

    # derived numerics
    h       = 1.0 / (ny - 1.0)
    width   = (nx - 1.0) / (ny - 1.0)
    dt_dif  = (opt.a_dif * min(h, h)^2) / max(opt.k, opt.Pr)

    # allocate field arrays
    S       = @zeros(nx, ny)
    vx      = @zeros(nx, ny)
    vy      = @zeros(nx, ny)
    v       = @zeros(nx, ny)
    T       = @zeros(nx, ny)
    dT2     = @zeros(nx, ny)
    dTx     = @zeros(nx, ny)
    dTy     = @zeros(nx, ny)
    T_rhs   = @zeros(nx, ny)
    W       = @zeros(nx, ny)
    dW2     = @zeros(nx, ny)
    dWx     = @zeros(nx, ny)
    dWy     = @zeros(nx, ny)
    W_rhs   = @zeros(nx, ny)
    Ra_dTdx = @zeros(nx, ny)

    init_array!(T, opt.T_init_strategy, h, width)
    init_array!(W, opt.W_init_strategy, h, width)

    # preallocate all the buffers necessary for multigrid
    prealloc_dict = preallocate_buffers(nx, ny)

    hx, hy = h, h

    # plotting
    T_storage = Vector{Matrix{Float64}}()
    W_storage = Vector{Matrix{Float64}}()
    S_storage = Vector{Matrix{Float64}}()

    tic = 0.0
    sim_time = 0.0
    step = 0
    while (sim_time < opt.ttot)
        if step == 3
            tic = time()
        end

        # solve for streamfunction S: D S = W (Dirichlet BCs = 0)
        r_rms = MGsolve_2DPoisson!(S, W, h, 0.0, opt.tol, opt.niters, false, prealloc_dict=prealloc_dict)

        # compute velocity field (vx, vy) from streamfunction S
        @parallel compute_velocity!(S, hx, hy, vx, vy)

        # compute velocity magnitude v
        v = sqrt.(vx.^2 + vy.^2)

        # compute timestep dt
        dt = compute_dt(v, vx, vy, dt_dif, opt.a_dif, opt.a_adv, h, opt.beta)

        # apply boundary conditions
        apply_boundary_conditions!(T)

        # compute Ra * dT / dx
        @parallel compute_Ra_dTdx!(opt.Ra, hx, T, Ra_dTdx)

        # diffusion terms for temperature T and vorticity W
        if (opt.beta ≉ 1.0)
            @parallel compute_diffusion2d!(T, hx, hy, opt.k, dT2)
            @parallel compute_diffusion2d!(W, hx, hy, opt.Pr, dW2)
        end

        # advection terms for temperature T and vorticity W
        @parallel compute_advection2d_x!(T, hx, vx, dTx)
        @parallel compute_advection2d_y!(T, hy, vy, dTy)
        @parallel compute_advection2d_x!(W, hx, vx, dWx)
        @parallel compute_advection2d_y!(W, hy, vy, dWy)

        # Euler step for temperature T and vorticity W
        if (opt.beta > 0.0)
            # semi-implicit step for temperature T
            c = 1.0 / (opt.beta * dt)
            T_rhs .= -c * ( T + dt * ( (1.0 - opt.beta) * dT2 - dTx - dTy ) )
            r_rms = MGsolve_2DPoisson!(T, T_rhs, h, c, opt.tol, opt.niters, true, prealloc_dict=prealloc_dict)

            # semi-implicit step for vorticity W
            c = c / opt.Pr
            W_rhs .= -c * ( W + dt * ( (1.0 - opt.beta) * dW2 - dWx - dWy - opt.Pr * Ra_dTdx ) )
            r_rms = MGsolve_2DPoisson!(W, W_rhs, h, c, opt.tol, opt.niters, false, prealloc_dict=prealloc_dict)
        else
            # explicit step for temperature T and vorticity W
            T .= T + dt * ( dT2 - dTx - dTy )
            W .= W + dt * ( dW2 - dWx - dWy - opt.Pr * Ra_dTdx )
        end

        sim_time += dt
        step += 1

        if ((step-1) % 20 == 0)
            if verbose
                println("time, step: $(sim_time) $(step)")
            end
            if do_vis
                push!(T_storage, copy(Array(T)))
                push!(W_storage, copy(Array(W)))
                push!(S_storage, copy(Array(S)))
            end
        end

        if testmode
            break
        end
    end
    toc = time()
    t_elapsed = toc - tic
    timed_iters = step - 3

    if do_vis
        save("/tmp/sim_results.jld", "T_storage", T_storage, "W_storage", W_storage, "S_storage", S_storage)
    end

    println("time, step: $(sim_time) $(step)")

    return SimOut_t(T, W, S, t_elapsed, timed_iters)
end

if abspath(PROGRAM_FILE) == @__FILE__
    opt = SimIn_t()
    opt.beta = 0.5
    opt.Pr = 1.0e-1
    opt.tol = 1.0e-7
    opt.nx = 1025
    opt.ny = 257
    navier_stokes_2D(opt=opt, do_vis=false)
end
