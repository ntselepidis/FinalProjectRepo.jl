using CUDA
import MPI
using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

macro qx(ix, iy, iz)
    esc(:(-D_dx * (Hτ[$ix + 1, $iy + 1, $iz + 1] - Hτ[$ix, $iy + 1, $iz + 1])))
end
macro qy(ix, iy, iz)
    esc(:(-D_dy * (Hτ[$ix + 1, $iy + 1, $iz + 1] - Hτ[$ix + 1, $iy, $iz + 1])))
end
macro qz(ix, iy, iz)
    esc(:(-D_dz * (Hτ[$ix + 1, $iy + 1, $iz + 1] - Hτ[$ix + 1, $iy + 1, $iz])))
end

@parallel_indices (ix, iy, iz) function diffusion_3D_step_τ(Ht, Hτ, Hτ2, dHdτ, dτ, _dt, _dx, _dy, _dz, D_dx, D_dy, D_dz)
    if (ix <= size(Hτ, 1) - 2 && iy <= size(Hτ, 2) - 2 && iz <= size(Hτ, 3) - 2)
        dHdτ[ix + 1, iy + 1, iz + 1] = (
            (@qx(ix + 1, iy, iz) - @qx(ix, iy, iz)) * _dx +
            (@qy(ix, iy + 1, iz) - @qy(ix, iy, iz)) * _dy +
            (@qz(ix, iy, iz + 1) - @qz(ix, iy, iz)) * _dz +
            (Hτ[ix + 1, iy + 1, iz + 1] - Ht[ix + 1, iy + 1, iz + 1]) * _dt
        )
        Hτ2[ix + 1, iy + 1, iz + 1] =
            Hτ[ix + 1, iy + 1, iz + 1] - dτ * dHdτ[ix + 1, iy + 1, iz + 1]
    end
    return nothing
end

@views function diffusion_3D_kernel_programming(; nx, ny, nz, do_vis = false, verbose=true, init_and_finalize_MPI=!isinteractive())
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
    Hτ2 = @zeros(nx, ny, nz)
    residual_H = @zeros(nx, ny, nz)

    H_g = zeros(nx * dims[1], ny * dims[2], nz * dims[3])

    _dt = 1.0 / dt
    _dx = 1.0 / dx
    _dy = 1.0 / dy
    _dz = 1.0 / dz
    D_dx = D / dx
    D_dy = D / dy
    D_dz = D / dz

    t = 0.0
    iter_outer = 0
    iter_total = 0
    while t < ttot

        iter_inner = 0
        err = 2 * tol
        while err > tol && iter_inner < iter_max
            @hide_communication (8, 8, 8) begin
                @parallel diffusion_3D_step_τ(Ht, Hτ, Hτ2, residual_H, dτ, _dt, _dx, _dy, _dz, D_dx, D_dy, D_dz)
                update_halo!(Hτ)
            end
            Hτ, Hτ2 = Hτ2, Hτ
            err = dist_norm_L2(residual_H * dt, comm_cart) / sqrt(total_N)
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

    X_g = LinRange(0 + dx / 2, lx - dx / 2, nx * dims[1])

    gather!(Array(Ht), H_g)

    finalize_global_grid(; finalize_MPI = init_and_finalize_MPI)

    return X_g, H_g
end
