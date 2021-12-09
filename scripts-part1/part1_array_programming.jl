using CUDA
import MPI
using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

include("part1_utils.jl")

@parallel function diffusion_3D_step_τ!(Ht, Hτ, dHdτ, dt, dτ, qx, qy, qz, dx, dy, dz, D)
    @all(qx) = D * @d_xi(Hτ) / dx
    @all(qy) = D * @d_yi(Hτ) / dy
    @all(qz) = D * @d_zi(Hτ) / dz

    @all(dHdτ) =
        -(@inn(Hτ) - @inn(Ht)) / dt + (@d_xa(qx) / dx + @d_ya(qy) / dy + @d_za(qz) / dz)
    @inn(Hτ) = @inn(Hτ) + @all(dHdτ) * dτ
    return nothing
end

@views function diffusion_3D_array_programming(; nx, ny, nz, do_vis = false, verbose=true, init_and_finalize_MPI=!isinteractive())
    # physics
    lx, ly, lz = 10.0, 10.0, 10.0
    D = 1.0
    ttot = 1.0

    # derived numerics
    me, dims, _nprocs, coords, comm_cart =
        init_global_grid(nx, ny, nz; init_MPI = init_and_finalize_MPI, quiet=!verbose)
    dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g()

    # bind MPI ranks to GPUs
    if USE_GPU
        select_device()
    end

    total_N = prod(dims) * nx * ny * nz

    # numerics
    dt = 0.2
    dτ = min(dx, dy, dz)^2 ./ D / 8.1
    nt = cld(ttot, dt)
    tol = 1e-8
    iter_max = 1e5

    # array allocation
    center = [lx / 2, ly / 2, lz / 2]
    qx = @zeros(nx - 1, ny - 2, nz - 2)
    qy = @zeros(nx - 2, ny - 1, nz - 2)
    qz = @zeros(nx - 2, ny - 2, nz - 1)
    Ht = @zeros(nx, ny, nz)
    Ht .= init_local_gaussian(center, dx, dy, dz, Ht)
    apply_boundary_conditions!(Ht, coords, dims)
    Hτ = copy(Ht)

    dHdt = @zeros(nx - 2, ny - 2, nz - 2)
    H_g = zeros(nx * dims[1], ny * dims[2], nz * dims[3])

    t = 0.0
    iter_outer = 0
    iter_total = 0
    while t < ttot

        iter_inner = 0
        err = 2 * tol
        while err > tol && iter_inner < iter_max
            @parallel diffusion_3D_step_τ!(Ht, Hτ, dHdt, dt, dτ, qx, qy, qz, dx, dy, dz, D)
            update_halo!(Hτ)
            err = dist_norm_L2(dHdt * dt, comm_cart) / sqrt(total_N)
            iter_inner += 1
        end
        if verbose && me == 0
            if err <= tol
                println("Converged after $iter_inner iterations.")
            else
                println("Couldn't converge within $iter_max iterations.")
            end
        end
        iter_total += iter_inner
        iter_outer += 1
        t += dt
        Ht .= Hτ

    end

    X_g = LinRange(0 + dx/2, lx - dx/2, nx * dims[1])

    gather!(Array(Ht), H_g)

    finalize_global_grid(; finalize_MPI = init_and_finalize_MPI)

    return X_g[2:end-1], H_g[2:end-1, 2:end-1, 2:end-1]
end
