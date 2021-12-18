# Part 2: another PDE
using SparseArrays
using SuiteSparse
using LinearAlgebra
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

@enum Init_t begin
    cosine
    random
end

@views function init_array!(M, scheme::Init_t, h, width)
    nx, ny = size(M)
    if scheme == cosine
        for i in 1:nx
            M[i, :] .= 0.5 * (1.0 + cos((3.0*π*(i-1)*h)/width))
        end
    elseif scheme == random
        M .= @rand(nx, ny)
    else
        error()
    end
end

@views function compute_dt(v, vx, vy, dt_dif, a_dif, a_adv, h, beta)
    v_max = maximum(v)
    if (v_max == 0)
        dt = dt_dif
    else
        # compute advective timestep
        dt_adv = a_adv * min(h / maximum(vx), h / maximum(vy))
        # compute timestep
        dt = (beta >= 0.5 ? dt_adv : min(dt_dif, dt_adv))
    end
    return dt
end

@views function stencil_5pt(nx, ny)
    Tx = spdiagm(-1 => 1 * @ones(nx - 1), 0 => -2 * @ones(nx), 1 => 1 * @ones(nx-1))
    Ty = spdiagm(-1 => 1 * @ones(ny - 1), 0 => -2 * @ones(ny), 1 => 1 * @ones(ny-1))
    Ix = spdiagm(0 => @ones(nx))
    Iy = spdiagm(0 => @ones(ny))
    T = kron(Ty, Ix) + kron(Iy, Tx)
    return T
end

@views function apply_boundary_conditions!(T)
    nx, ny = size(T)
    T[:, 1]  .= 1.0
    T[:, ny] .= 0.0
    T[1, :]  .= T[2, :]
    T[nx, :] .= T[nx-1, :]
end

@views function navier_stokes_2D()
    # physics
    k       = 1.0     # thermal diffusivity
    Ra      = 1.0e6   # Rayleigh number
    Pr      = 1.0e-2  # Prandtl number

    # numerics
    nx, ny  = 257, 65 # number of grid-points in x- and y-directions
    ttot    = 0.1     # total time for simulation
    #beta    = 0.5     # explicit/implicit/semi-implicit
    beta    = 0.0     # explicit/implicit/semi-implicit
    niters  = 50      # max number of iterations for iterative solver
    tol     = 1.0e-3  # tolerance for linear solver
    a_dif   = 0.15    # diffusive timestep parameter
    a_adv   = 0.4     # advective timestep parameter
    Tinit   = cosine

    # derived numerics
    h       = 1.0 / (ny - 1.0)
    width   = (nx - 1.0) / (ny - 1.0)
    dt_dif  = (a_dif * min(h, h)^2) / max(k, Pr)

    # allocate field arrays
    S       = @zeros(nx, ny)
    vx      = @zeros(nx, ny)
    vy      = @zeros(nx, ny)
    v       = @zeros(nx, ny)
    T       = @zeros(nx, ny)
    dT2     = @zeros(nx, ny)
    dTx     = @zeros(nx, ny)
    dTy     = @zeros(nx, ny)
    Trhs    = @zeros(nx, ny)
    W       = @zeros(nx, ny)
    dW2     = @zeros(nx, ny)
    dWx     = @zeros(nx, ny)
    dWy     = @zeros(nx, ny)
    Wrhs    = @zeros(nx, ny)
    Ra_dTdx = @zeros(nx, ny)

    init_array!(T, Tinit, h, width)
    init_array!(W, random, h, width)

    # create coefficient matrix A from 5 pt stencil
    A = stencil_5pt(nx, ny) / h^2

    # compute ldlt factorization
    A_chol = ldlt(A)

    hx, hy = h, h

    time = 0.0
    step = 0
    while (time < ttot)

        # solve for stream function S: D S = W (Dirichlet BCs = 0)
        # r_rms = MGsolve_2DPoisson(S, W, h, 0.0, err, niters, .false.)
        S[:] .= A_chol \ W[:]

        # compute velocity field (vx, vy) from stream function S
        @parallel_indices (ix, iy) function compute_velocity!(S, hx, hy, vx, vy)
            if (1 < ix < size(S, 1) && 1 < iy < size(S, 2))
                vx[ix, iy] =  ( S[ix, iy+1] - S[ix, iy-1] ) / ( 2 * hy )
                vy[ix, iy] = -( S[ix+1, iy] - S[ix-1, iy] ) / ( 2 * hx )
            end
            return nothing
        end
        @parallel compute_velocity!(S, hx, hy, vx, vy)

        # compute velocity magnitude v
        v = sqrt.(vx.^2 + vy.^2)

        # compute timestep dt
        dt = compute_dt(v, vx, vy, dt_dif, a_dif, a_adv, h, beta)

        # apply boundary conditions
        apply_boundary_conditions!(T)

        # compute Ra * dT / dx
        @parallel_indices (ix, iy) function compute_Ra_dTdx!(Ra, hx, T, Ra_dTdx)
            if (1 < ix < size(S, 1) && 1 < iy < size(S, 2))
                Ra_dTdx[ix, iy] =  ( T[ix+1, iy] - T[ix-1, iy] ) / ( 2 * hx )
            end
            return nothing
        end
        @parallel compute_Ra_dTdx!(Ra, hx, T, Ra_dTdx)

        if (beta ≉ 1.0)
            @parallel_indices (ix, iy) function compute_diffusion2d!(T, hx, hy, k, dT2)
                if (1 < ix < size(T, 1) && 1 < iy < size(T, 2))
                    dT2[ix, iy] = k * (( T[ix+1, iy] - 2*T[ix, iy]+ T[ix-1, iy] ) / hx^2 +
                                       ( T[ix, iy+1] - 2*T[ix, iy]+ T[ix, iy-1] ) / hy^2)
                end
                return nothing
            end
            @parallel compute_diffusion2d!(T, hx, hy, k, dT2)
            @parallel compute_diffusion2d!(W, hx, hy, Pr, dW2)
        end

        # advection terms for temperature T and vorticity W
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
        @parallel compute_advection2d_x!(T, hx, vx, dTx)
        @parallel compute_advection2d_x!(W, hx, vx, dWx)

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
        @parallel compute_advection2d_y!(T, hy, vy, dTy)
        @parallel compute_advection2d_y!(W, hy, vy, dWy)

        # Euler step for temperature T and vorticity W
        if (beta > 0.0)
            # semi-implicit step for temperature T
            # c = 1.0 / (beta * dt)
            # T_rhs = -c * ( T + dt * ( (1.0 - beta) * dT2 - dTx - dTy ) )
            # r_rms = MGsolve_2DPoisson(T, T_rhs, h, c, err, niters, .true.)

            # semi-implicit step for vorticity W
            # c = c / Pr
            # W_rhs = -c * ( W + dt * ( (1.0 - beta) * dW2 - dWx - dWy - Pr * Ra_dTdx ) )
            # r_rms = MGsolve_2DPoisson(W, W_rhs, h, c, err, niters, .false.)
        else
            # explicit step for temperature T and vorticity W
            T .= T + dt * ( dT2 - dTx - dTy )
            W .= W + dt * ( dW2 - dWx - dWy - Pr * Ra_dTdx )
        end

        time += dt
        step += 1

        if (step % 20 == 0)
            println("time, step: $(time), $(step)")
        end
    end

    println("time, step: $(time), $(step)")

end

navier_stokes_2D()
