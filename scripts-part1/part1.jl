# Part 1 of final project: Diffusion equation
USE_GPU = length(ARGS) > 0 && ARGS[1] == "gpu"
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

function init_local_gaussian(center, dx, dy, dz, H)
    X = Data.Array([
        2 * exp(
            -1.0 * (
                (x_g(ix, dx, H) + dx / 2 - center[1])^2 +
                (y_g(iy, dy, H) + dy / 2 - center[2])^2 +
                (z_g(iz, dz, H) + dz / 2 - center[3])^2
            ),
        ) for ix = 1:size(H, 1), iy = 1:size(H, 2), iz = 1:size(H, 3)
    ])
    return X
end

function apply_boundary_conditions!(H, coords, dims)
    if coords[1] == 1
        H[1, :, :] .= 0.0
    end
    if coords[2] == 1
        H[:, 1, :] .= 0.0
    end
    if coords[3] == 1
        H[:, :, 1] .= 0.0
    end

    if coords[1] == dims[1]
        H[end, :, :] .= 0.0
    end
    if coords[2] == dims[2]
        H[:, end, :] .= 0.0
    end
    if coords[3] == dims[3]
        H[:, :, end] .= 0.0
    end
end

@parallel function diffusion_3D_step_τ!(Ht, Hτ, dHdτ, dt, dτ, qx, qy, qz, dx, dy, dz, D)
    @all(qx) = D * @d_xi(Hτ) / dx
    @all(qy) = D * @d_yi(Hτ) / dy
    @all(qz) = D * @d_zi(Hτ) / dz

    @all(dHdτ) =
        -(@inn(Hτ) - @inn(Ht)) / dt + (@d_xa(qx) / dx + @d_ya(qy) / dy + @d_za(qz) / dz)
    @inn(Hτ) = @inn(Hτ) + @all(dHdτ) * dτ
    return nothing
end

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

function dist_norm_L2(Rh, comm_cart)
    sq_residual = [sum(Rh .^ 2)]  # this is local
    MPI.Allreduce!(sq_residual, +, comm_cart)
    return sqrt(sq_residual[1])
end

@views function diffusion_3D(; nx, ny, nz, do_vis = false)
    # physics
    lx, ly, lz = 10.0, 10.0, 10.0
    D = 1.0
    ttot = 1.0

    # derived numerics
    me, dims, _nprocs, coords, comm_cart =
        init_global_grid(nx, ny, nz; init_MPI = !isinteractive())
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
        if me == 0
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

    finalize_global_grid(; finalize_MPI = !isinteractive())

    return X_g[2:(end - 1)], H_g[2:(end - 1), 2:(end - 1), 2:(end - 1)]
end

if abspath(PROGRAM_FILE) == @__FILE__
    local nx_, ny_, nz_ = 32, 32, 32
    if length(ARGS) > 1
        try
            nx_, ny_, nz_ = parse.(Int, ARGS[2:4])
        catch
            println("Usage: mpiexecjl -np <np> julia --project scripts/part1.jl [cpu/gpu] [nx ny nz]")
            exit()
        end
    end
    diffusion_3D(nx = nx_, ny = ny_, nz = nz_)
end
