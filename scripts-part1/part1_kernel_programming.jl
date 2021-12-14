using CUDA
import MPI
using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using BenchmarkTools
using Printf
using ProgressBars

macro qx(ix, iy, iz)
    esc(:(-D_dx * (Hτ[$ix, $iy, $iz] - Hτ[$ix-1, $iy, $iz])))
end
macro qy(ix, iy, iz)
    esc(:(-D_dy * (Hτ[$ix, $iy, $iz] - Hτ[$ix, $iy-1, $iz])))
end
macro qz(ix, iy, iz)
    esc(:(-D_dz * (Hτ[$ix, $iy, $iz] - Hτ[$ix, $iy, $iz-1])))
end

""" Kernelized 3D diffusion step

Cells         = (nx-2) * (ny-2) * (nz-2)

Work per cell = 3 * (3 * sub + 3 * mul) + 1 * (1 * sub + 1 * mul) + 3 * add + 1 * (1 * sub + 1 * mul)
              = 25 FLOP
Work          = 25 FLOP * Cells

Memory moved per cell  = 9 * read Hτ + 1 * read Ht + 1 * read/write dHdτ + 1 * read/write Hτ2
                       = 14 Float64 read or write 
Memory moved           = 14 Float64 * Cells

Note that we can probably reduce the memory loaded by a lot using shard memory
"""
@parallel_indices (ix, iy, iz) function diffusion_3D_step_τ(Ht, Hτ, Hτ2, dHdτ, dτ, _dt, _dx, _dy, _dz, D_dx, D_dy, D_dz)
    if (1 < ix < size(Hτ, 1) && 1 < iy < size(Hτ, 2) && 1 < iz < size(Hτ, 3))
        dHdτ[ix, iy, iz] = (  # We read full stencil from Hτ ( 9 elements in 3D ) and write once
            (@qx(ix + 1, iy, iz) - @qx(ix, iy, iz)) * _dx +  # Work = 3 * sub + 3 * mul
            (@qy(ix, iy + 1, iz) - @qy(ix, iy, iz)) * _dy +
            (@qz(ix, iy, iz + 1) - @qz(ix, iy, iz)) * _dz +
            (Hτ[ix, iy, iz] - Ht[ix, iy, iz]) * _dt  # Work = 1 * sub + 1 * mul
        )
        Hτ2[ix, iy, iz] =
          Hτ[ix, iy, iz] - dτ * dHdτ[ix, iy, iz]  # Work = 1 * sub + 1 * mul (or 1 * fma)
    end
    return nothing
end

""" Kernelized 3D diffusion step with shared memory

Similar to `diffusion_3D_step_τ`, but massively reducing memory loads by using shared memory (on GPU).

Cells         = (nx-2) * (ny-2) * (nz-2)

Work per cell = 3 * (3 * sub + 3 * mul) + 1 * (1 * sub + 1 * mul) + 3 * add + 1 * (1 * sub + 1 * mul)
              = 25 FLOP
Work          = 25 FLOP * Cells

Memory moved per cell  = 1 * read Hτ + 1 * read Ht + 1 * read/write dHdτ + 1 * read/write Hτ2
                       = 6 Float64 read or write 
Memory moved           = 6 Float64 * Cells

Note that we can probably reduce the memory loaded by a lot using shard memory
"""
@parallel_indices (ix, iy, iz) function diffusion_3D_step_τ_shared_memory(Ht, Hτ_, Hτ2, dHdτ, dτ, _dt, _dx, _dy, _dz, D_dx, D_dy, D_dz)
    tx, ty, tz = @threadIdx().x + 1, @threadIdx().y + 1, @threadIdx().z + 1
    Hτ = @sharedMem(eltype(Ht), (@blockDim().x+2, @blockDim().y+2, @blockDim().z+2))
    Hτ[tx, ty, tz] = Hτ_[ix, iy, iz]
    if (1 < ix < size(Hτ_, 1) && 1 < iy < size(Hτ_, 2) && 1 < iz < size(Hτ_, 3))
        if (@threadIdx().x == 1)             Hτ[tx-1,ty,tz] = Hτ_[ix-1,iy,iz] end
        if (@threadIdx().x == @blockDim().x) Hτ[tx+1,ty,tz] = Hτ_[ix+1,iy,iz] end
        if (@threadIdx().y == 1)             Hτ[tx,ty-1,tz] = Hτ_[ix,iy-1,iz] end
        if (@threadIdx().y == @blockDim().y) Hτ[tx,ty+1,tz] = Hτ_[ix,iy+1,iz] end
        if (@threadIdx().z == 1)             Hτ[tx,ty,tz-1] = Hτ_[ix,iy,iz-1] end
        if (@threadIdx().z == @blockDim().z) Hτ[tx,ty,tz+1] = Hτ_[ix,iy,iz+1] end
        @sync_threads()
        dHdτ[ix, iy, iz] = (  # We read full stencil from Hτ ( 9 elements in 3D ) and write once
            (@qx(tx + 1, ty, tz) - @qx(tx, ty, tz)) * _dx +  # Work = 3 * sub + 3 * mul
            (@qy(tx, ty + 1, tz) - @qy(tx, ty, tz)) * _dy +
            (@qz(tx, ty, tz + 1) - @qz(tx, ty, tz)) * _dz +
            (Hτ_[ix, iy, iz] - Ht[ix, iy, iz]) * _dt  # Work = 1 * sub + 1 * mul
        )
        Hτ2[ix, iy, iz] =
          Hτ_[ix, iy, iz] - dτ * dHdτ[ix, iy, iz]  # Work = 1 * sub + 1 * mul (or 1 * fma)
    end
    return nothing
end

@views function diffusion_3D_kernel_programming(; nx, ny, nz, ttot = 1., use_shared_memory=true, do_vis = false, verbose=true, init_and_finalize_MPI=!isinteractive())
    # physics
    lx, ly, lz = 10 .* (nx, ny, nz).÷32
    D = 1.0

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

    iter_outer = 0
    timed_iter_total = 0  # without warmup
    tic = 0

    # GPU shared_memory setup
    if use_shared_memory
        threads = (32, 8, 1)  # these could be fined tuned still
        blocks  = (nx, ny, nz) .÷ threads
        shmem = prod(threads.+2)*sizeof(Float64) 
    end

    for t ∈ ProgressBar(0:dt:ttot-dt)
        if iter_outer == 3
            tic = time()
        end
        iter_inner = 0
        err = 2 * tol
        while err > tol && iter_inner < iter_max
            if use_shared_memory
                @parallel blocks threads shmem=shmem diffusion_3D_step_τ_shared_memory(Ht, Hτ, Hτ2, residual_H, dτ, _dt, _dx, _dy, _dz, D_dx, D_dy, D_dz)
            else
                # currently `@hide_communication` doesn't work with shared memory
                @hide_communication (8, 8, 8) begin
                    @parallel diffusion_3D_step_τ(Ht, Hτ, Hτ2, residual_H, dτ, _dt, _dx, _dy, _dz, D_dx, D_dy, D_dz)
                    update_halo!(Hτ)
                end
            end
            Hτ, Hτ2 = Hτ2, Hτ  # pointer swap, i.e. no memory movement
            err = dist_norm_L2(residual_H * dt, comm_cart) / sqrt(total_N)  # ≈ (1 * read & 1 * pow + 1 * add) * (nx-2) * (ny-2) * (nz-2)
            iter_inner += 1
        end
        if verbose && me == 0
            if err <= tol
                println("Converged after $iter_inner iterations.")
            else
                println("Couldn't converge within $iter_max iterations.")
            end
        end
        timed_iter_total += iter_inner
        iter_outer += 1
        Ht .= Hτ  # ≈ 1 * read/write + 1 * read * (nx-2) * (ny-2) * (nz-2), but these are "few" compared to the inner iter
    end
    toc = time()
    Δt = toc - tic  # seconds

    Work = timed_iter_total * (25+2) * (nx-2) * (ny-2) * (nz-2)  # see kernel docstring
    Performance = Work / Δt
    Memory = timed_iter_total * (6+1) * sizeof(Float64) * (nx-2) * (ny-2) * (nz-2)
    Intensity = Work / Memory
    BenchResults = @NamedTuple{Work::Int, Performance::Float64, Memory::Int, Intensity::Float64}


    println("Finished after $(iter_outer-2) outer iterations in $(@sprintf("%3.3f", Δt)) seconds of compute!")

    X_g = LinRange(0 + dx / 2, lx - dx / 2, nx * dims[1])

    gather!(Array(Ht), H_g)

    finalize_global_grid(; finalize_MPI = init_and_finalize_MPI)

    return X_g, H_g, BenchResults((Work=Work, Performance=Performance, Memory=Memory, Intensity=Intensity))
end
