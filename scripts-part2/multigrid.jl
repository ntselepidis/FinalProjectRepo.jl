using LinearAlgebra
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using CUDA

include("part2_utils.jl")

@enum ExecutionPolicy_t begin
    serial          # uses Gauss-Seidel smoother and coarse solver, and no ParallelStencil code at all
    parallel        # uses Jacobi smoother and coarse solver, and ParallelStencil code
    parallel_shmem  # uses Jacobi smoother and coarse solver, and ParallelStencil code + shared memory for residual_2DPoisson
end

mutable struct MGOpt
    coarse_solve_size :: Int
    execution_policy :: ExecutionPolicy_t

    MGOpt() = new(5, parallel_shmem)
end

# solves $(\nabla^2 - c) u = f$ using V-cycle multigrid
function MGsolve_2DPoisson!(u::AbstractArray{Float64}, f::AbstractArray{Float64}, h::Float64, c::Float64, tol::Float64, niters::Int, apply_BCs::Bool; opt=MGOpt(), verbose=false)
    # set nx, ny
    nx, ny = size(u)

    @assert opt.coarse_solve_size <= min(nx, ny)
    @assert isinteger(log2(opt.coarse_solve_size - 1))

    f_rms = sqrt(sum(f.^2)/(nx*ny))

    tolf = tol * f_rms

    r_rms = 0.0
    for iter = 1:niters
        # apply Dirichlet and Neumann BCs for temperature T
        if (apply_BCs)
            apply_boundary_conditions!(u)
        end
        # execute V-cycle iteration
        r_rms = Vcycle_2DPoisson!(u, f, h, c, tol, opt.coarse_solve_size, opt.execution_policy, apply_BCs)
        @synchronize()

        if verbose
            println("$(iter) $(r_rms / f_rms)")
        end
        if (r_rms < tolf)
            if verbose
                println("V-cycle multigrid converged in $(iter) iterations.")
            end
            break
        end
    end

    if (r_rms > tolf)
        @warn "V-cycle multigrid failed to converge within" niters "iterations."
    end

    return r_rms

end

"""

nx, ny must be = (2^k)+1

"""
function Vcycle_2DPoisson!(u_f::AbstractArray{Float64}, rhs::AbstractArray{Float64}, h::Float64, c::Float64, tol::Float64, coarse_solve_size::Int, execution_policy::ExecutionPolicy_t, apply_BCs::Bool)

    nx, ny = size(u_f)

    if ( nx-1 != 2*((nx-1)÷2) || ny-1 != 2*((ny-1)÷2) )
        error("ERROR:not a power of 2")
    end

    # coarse grid size
    nxc = 1+(nx-1)÷2
    nyc = 1+(ny-1)÷2

    res_rms = 0.
    if (min(nx, ny) > coarse_solve_size)   # not the coarsest level

        res_f = @zeros(nx, ny)
        corr_f = @zeros(nx, ny)
        corr_c = @zeros(nxc, nyc)
        res_c = @zeros(nxc, nyc)

        #---------- take 2 iterations on the fine grid--------------
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, execution_policy)
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, execution_policy)

        #--------- restrict the residual to the coarse grid --------
        residual_2DPoisson_wrapper!(u_f, rhs, h, c, res_f, execution_policy)
        restrict_wrapper!(res_f, res_c, apply_BCs, execution_policy)

        #---------- solve for the coarse grid correction -----------
        corr_c .= 0.
        res_rms = Vcycle_2DPoisson!(corr_c, res_c, h*2, c, tol, coarse_solve_size, execution_policy, apply_BCs) # *RECURSIVE CALL*

        #---- prolongate (interpolate) the correction to the fine grid
        prolongate_wrapper!(corr_c, corr_f, apply_BCs, execution_policy)

        #---------- correct the fine-grid solution -----------------
        u_f .= u_f - corr_f

        #---------- two more smoothing iterations on the fine grid---
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, execution_policy)
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, execution_policy)

    else

        #----- coarsest level (ny=5): iterate to get 'exact' solution
        coarse_solve_iters = 20*coarse_solve_size
        tol_rhs = tol * sqrt(sum(rhs.^2)/(nx*ny))
        for i = 1:coarse_solve_iters  # heuristic
            res_rms = iteration_2DPoisson!(u_f, rhs, h, c, execution_policy)
            if res_rms < tol_rhs
                break
            end
        end
        if res_rms > tol_rhs
            @debug "Coarse solve of V-cycle multigrid did not converge to " tol " within " coarse_solve_iters "iterations. This is usually fine as long as it converges in a fixed amount of Vcycles."
        end

    end

    return res_rms   # returns the rms. residual
end

# computes the residual $R = (\nabla^2 - c) u - f$ in array res
@parallel_indices (ix, iy) function residual_2DPoisson!(u::AbstractArray{Float64}, f::AbstractArray{Float64}, h::Float64, c::Float64, res::AbstractArray{Float64})
    nx, ny = size(u)
    C = (4.0 + c * h^2)
    _h2 = 1 / h^2
    if (1 < ix < nx && 1 < iy < ny)
        res[ix, iy] = (
                       (  u[ix+1, iy]
                        + u[ix-1, iy]
                        + u[ix, iy+1]
                        + u[ix, iy-1]
                        - C * u[ix, iy] ) * _h2
                       - f[ix, iy]
                      )
    end
    return nothing
end

@parallel_indices (ix, iy) function residual_2DPoisson_shmem!(u_::AbstractArray{Float64}, f::AbstractArray{Float64}, h::Float64, c::Float64, res::AbstractArray{Float64})
    nx, ny = size(u_)
    C = (4.0 + c * h^2)
    _h2 = 1 / h^2

    tx = @threadIdx().x + 1
    ty = @threadIdx().y + 1
    u = @sharedMem(eltype(u_), (@blockDim().x+2, @blockDim().y+2))
    if (1 <= ix <= nx && 1 <= iy <= ny)
        u[tx, ty] = u_[ix, iy]
    end
    @sync_threads()
    if (1 < ix < nx && 1 < iy < ny)
        u[tx, ty] = u_[ix, iy]
        if (@threadIdx().x == 1)             u[tx-1, ty] = u_[ix-1, iy] end
        if (@threadIdx().x == @blockDim().x) u[tx+1, ty] = u_[ix+1, iy] end
        if (@threadIdx().y == 1)             u[tx, ty-1] = u_[ix, iy-1] end
        if (@threadIdx().y == @blockDim().y) u[tx, ty+1] = u_[ix, iy+1] end
        @sync_threads()
        res[ix, iy] = (
                       (  u[tx+1, ty]
                        + u[tx-1, ty]
                        + u[tx, ty+1]
                        + u[tx, ty-1]
                        - C * u[tx, ty] ) * _h2
                       - f[ix, iy]
                      )
        @sync_threads()
    end
    return nothing
end

function residual_2DPoisson_wrapper!(u_f, rhs, h, c, res_f, execution_policy)
    nx, ny = size(u_f)

    if execution_policy == parallel_shmem
        threads = (32, 8)
        blocks = ((nx, ny) .- 1) .÷ (threads) .+ 1
        shmem = prod(threads.+2) * sizeof(Float64)
        @parallel blocks threads shmem=shmem residual_2DPoisson_shmem!(u_f, rhs, h, c, res_f)
    elseif execution_policy == parallel
        @parallel residual_2DPoisson!(u_f, rhs, h, c, res_f)
    elseif execution_policy == serial
        error() # Not implemented yet
    else
        error()
    end
end

# performs one Jacobi iteration on field u
# returns rms residual
function iteration_2DPoisson!(u, f, h, c, execution_policy; alpha=4.0/5.0)
    nx, ny = size(u)

    # Do one Jacobi iteration
    res = @zeros(nx, ny)  # TODO preallocate this
    residual_2DPoisson_wrapper!(u, f, h, c, res, execution_policy)

    # update r_rms
    r_rms = sqrt(sum(res.^2) / (nx * ny))

    # update u
    u .+= alpha * (h^2 / (4.0 + c * h^2)) .* res

    return r_rms
end

""" TODO: Note something about alpha """
@views function iteration_2DPoisson_gs!(u, f, h, c; alpha = 1.0)

    nx, ny = size(u)

    # Gauss-Seidel iteration
    r_rms = 0.0
    for j = 2: ny - 1
        for i = 2: nx - 1

            # compute residual
            r = ( u[i+1, j]
                 + u[i-1, j]
                 + u[i, j+1]
                 + u[i, j-1]
                 - (4.0 + c * h^2) * u[i, j] ) / h^2 - f[i, j]

            # update u
            u[i, j] = u[i, j] + alpha * (h^2 / (4.0 + c * h^2)) * r

            # update r_rms
            r_rms = r_rms + r^2

        end
    end

    r_rms = sqrt(r_rms / (nx * ny))

    return r_rms
end

@parallel_indices (ix, iy) function restrict!(fine, coarse)
    nx, ny = size(fine)
    if (ix % 2 == 1 && iy % 2 == 1) && (3 <= ix <= nx-2 && 3 <= iy <= ny-2)
        ix_c, iy_c = ((ix-1)÷2)+1, ((iy-1)÷2)+1
        coarse[ix_c, iy_c] = fine[ix, iy]
    end
    return nothing
end

@views function restrict_wrapper!(fine, coarse, apply_BCs, execution_policy)
    # initalize coarse to zero (+ Dirichlet(0) BCs)
    coarse[:] .= 0.0

    if execution_policy in [parallel, parallel_shmem]
        @parallel restrict!(fine, coarse)
    else
        restrict_serial!(fine, coarse)
    end

    # apply Neumann BCs for temperature T
    if apply_BCs
        apply_boundary_conditions_neumann!(coarse)
    end
end

@parallel_indices (ix, iy) function prolongate_with_atomic!(coarse, fine)
    nx, ny = size(fine)
    a2 = 1.0 / 2.0
    a4 = 1.0 / 4.0
    if (ix % 2 == 1 && iy % 2 == 1) && (3 <= ix <= nx-2 && 3 <= iy <= ny-2)
        ix_c, iy_c = ((ix-1)÷2)+1, ((iy-1)÷2)+1
        @atomic fine[ix, iy] = fine[ix, iy] + coarse[ix_c, iy_c]
        @atomic fine[ix+1, iy] = fine[ix+1, iy] + a2 * coarse[ix_c, iy_c]
        @atomic fine[ix-1, iy] = fine[ix-1, iy] + a2 * coarse[ix_c, iy_c]
        @atomic fine[ix, iy+1] = fine[ix, iy+1] + a2 * coarse[ix_c, iy_c]
        @atomic fine[ix, iy-1] = fine[ix, iy-1] + a2 * coarse[ix_c, iy_c]
        @atomic fine[ix+1, iy+1] = fine[ix+1, iy+1] + a4 * coarse[ix_c, iy_c]
        @atomic fine[ix+1, iy-1] = fine[ix+1, iy-1] + a4 * coarse[ix_c, iy_c]
        @atomic fine[ix-1, iy+1] = fine[ix-1, iy+1] + a4 * coarse[ix_c, iy_c]
        @atomic fine[ix-1, iy-1] = fine[ix-1, iy-1] + a4 * coarse[ix_c, iy_c]
    end
    return nothing
end

@parallel_indices (ix, iy) function prolongate!(coarse, fine)
    nx, ny = size(fine)
    a2 = 1.0 / 2.0
    a4 = 1.0 / 4.0
    if (ix % 2 == 1 && iy % 2 == 1) && (3 <= ix <= nx-2 && 3 <= iy <= ny-2)
        ix_c, iy_c = ((ix-1)÷2)+1, ((iy-1)÷2)+1
        fine[ix, iy] = fine[ix, iy] + coarse[ix_c, iy_c]
        fine[ix+1, iy] = fine[ix+1, iy] + a2 * coarse[ix_c, iy_c]
        fine[ix-1, iy] = fine[ix-1, iy] + a2 * coarse[ix_c, iy_c]
        fine[ix, iy+1] = fine[ix, iy+1] + a2 * coarse[ix_c, iy_c]
        fine[ix, iy-1] = fine[ix, iy-1] + a2 * coarse[ix_c, iy_c]
        fine[ix+1, iy+1] = fine[ix+1, iy+1] + a4 * coarse[ix_c, iy_c]
        fine[ix+1, iy-1] = fine[ix+1, iy-1] + a4 * coarse[ix_c, iy_c]
        fine[ix-1, iy+1] = fine[ix-1, iy+1] + a4 * coarse[ix_c, iy_c]
        fine[ix-1, iy-1] = fine[ix-1, iy-1] + a4 * coarse[ix_c, iy_c]
    end
    return nothing
end

@views function prolongate_wrapper!(coarse, fine, apply_BCs, execution_policy)
    # initialize fine to zero (+ Dirichlet(0) BCs)
    fine[:] .= 0.0

    # We need the @atomic macro to avoid race-conditions on the GPU,
    # but this is not supported for CPU :(
    if execution_policy in [parallel, parallel_shmem]
        if USE_GPU
            @parallel prolongate_with_atomic!(coarse, fine)
        else
            @parallel prolongate!(coarse, fine)
        end
    else
        prolongate_serial!(coarse, fine)
    end

    # apply Neumann BCs for temperature T
    if apply_BCs
        apply_boundary_conditions_neumann!(fine)
    end

end

@views function prolongate_serial!(coarse, fine, apply_BCs)
    a2, a4 = 1.0 / 2.0, 1.0 / 4.0

    nx = size(fine, 1)
    ny = size(fine, 2)

    # initialize fine to zero (+ Dirichlet(0) BCs)
    fine .= 0.0

    for j = 1:ny
        for i = 1: nx
            ic, jc = ((i-1)÷2)+1, ((j-1)÷2)+1
            if (i % 2 == 1 && j % 2 == 1) && (3 <= i <= nx-2 && 3 <= j <= ny-2)
                fine[i, j] = coarse[ic, jc]
                fine[i+1, j] = fine[i+1, j] + a2 * coarse[ic, jc]
                fine[i-1, j] = fine[i-1, j] + a2 * coarse[ic, jc]
                fine[i, j+1] = fine[i, j+1] + a2 * coarse[ic, jc]
                fine[i, j-1] = fine[i, j-1] + a2 * coarse[ic, jc]
                fine[i+1, j+1] = fine[i+1, j+1] + a4 * coarse[ic, jc]
                fine[i+1, j-1] = fine[i+1, j-1] + a4 * coarse[ic, jc]
                fine[i-1, j+1] = fine[i-1, j+1] + a4 * coarse[ic, jc]
                fine[i-1, j-1] = fine[i-1, j-1] + a4 * coarse[ic, jc]
            end
        end
    end

    # apply Neumann BCs for temperature T
    if (apply_BCs)
        apply_neumann_boundary_conditions(fine)
    end

end

@views function  restrict_serial!(fine, coarse, apply_BCs)
    nx = size(fine, 1)
    ny = size(fine, 2)

    # initalize coarse to zero (+ Dirichlet(0) BCs)
    coarse .= 0.0

    # apply restriction stencil
    jc = 2
    for j = 3:2: ny-2
        ic = 2
        for i = 3:2: nx-2
            coarse[ic, jc] = fine[i, j]
            ic = ic + 1
        end
        jc = jc + 1
    end

    # apply Neumann BCs for temperature T
    if (apply_BCs)
        apply_neumann_boundary_conditions(coarse)
    end

end
