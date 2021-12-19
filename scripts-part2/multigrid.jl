using LinearAlgebra
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
#if !ParallelStencil.is_initialized()
#    @init_parallel_stencil(Threads, Float64, 2)
#end

include("part2_utils.jl")

# solves $(\nabla^2 - c) u = f$ using V-cycle multigrid
function MGsolve_2DPoisson!(u::AbstractArray{Float64}, f::AbstractArray{Float64}, h::Float64, c::Float64, tol::Float64, niters::Int, apply_BCs::Bool)
    # set nx, ny
    nx, ny = size(u)

    f_rms = sqrt(sum(f.^2)/(nx*ny))

    tolf = tol * f_rms

    r_rms = 0.0
    for iter = 1:niters
        # apply Dirichlet and Neumann BCs for temperature T
        if (apply_BCs)
            apply_boundary_conditions!(u)
        end
        # execute V-cycle iteration
        r_rms = Vcycle_2DPoisson!(u, f, h, c, apply_BCs)

        # println("$(iter) $(r_rms / f_rms)")
        if (r_rms < tolf)
            # println("V-cycle multigrid converged in $(iter) iterations.")
            break
        end
    end

    if (r_rms > tolf)
        @warn "V-cycle multigrid failed to converge within" niters "iterations."
    end

end

"""

nx, ny must be = (2^k)+1

"""
function Vcycle_2DPoisson!(u_f::AbstractArray{Float64}, rhs::AbstractArray{Float64}, h::Float64, c::Float64, apply_BCs::Bool)

    # use alpha = 1.0 for Gauss-Seidel
    # and alpha = 4/5 for Jacobi

    alpha = 4.0/5.0

    nx, ny = size(u_f)

    if ( nx-1 != 2*((nx-1)÷2) || ny-1 != 2*((ny-1)÷2) )
        error("ERROR:not a power of 2")
    end

    # coarse grid size
    nxc = 1+(nx-1)÷2
    nyc = 1+(ny-1)÷2

    if (min(nx, ny) > 5)   # not the coarsest level

        res_f = @zeros(nx, ny)
        corr_f = @zeros(nx, ny)
        corr_c = @zeros(nxc, nyc)
        res_c = @zeros(nxc, nyc)

        #---------- take 2 iterations on the fine grid--------------
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, alpha)
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, alpha)

        #--------- restrict the residual to the coarse grid --------
        @parallel residual_2DPoisson!(u_f, rhs, h, c, res_f)
        restrict_wrapper!(res_f, res_c, apply_BCs)

        #---------- solve for the coarse grid correction -----------
        corr_c .= 0.
        res_rms = Vcycle_2DPoisson!(corr_c, res_c, h*2, c, apply_BCs) # *RECURSIVE CALL*

        #---- prolongate (interpolate) the correction to the fine grid
        prolongate_wrapper!(corr_c, corr_f, apply_BCs)

        #---------- correct the fine-grid solution -----------------
        u_f .= u_f - corr_f

        #---------- two more smoothing iterations on the fine grid---
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, alpha)
        res_rms = iteration_2DPoisson!(u_f, rhs, h, c, alpha)

    else

        #----- coarsest level (ny=5): iterate to get 'exact' solution
        for i = 1:100
            res_rms = iteration_2DPoisson!(u_f, rhs, h, c, alpha)
        end

    end

    resV = res_rms   # returns the rms. residual

end

# computes the residual $R = (\nabla^2 - c) u - f$ in array res
@parallel_indices (ix, iy) function residual_2DPoisson!(u::AbstractArray{Float64}, f::AbstractArray{Float64}, h::Float64, c::Float64, res::AbstractArray{Float64})
    nx, ny = size(u)
    if (1 < ix < nx && 1 < iy < ny)
        res[ix, iy] = (
                       (  u[ix+1, iy]
                        + u[ix-1, iy]
                        + u[ix, iy+1]
                        + u[ix, iy-1]
                        - (4.0 + c * h^2) * u[ix, iy] ) / h^2
                       - f[ix, iy]
                      )
    end
    return nothing
end

@parallel function update_u!(res, C, u)
    @all(u) = @all(u) + C .* @all(res)
    return nothing
end

# performs one Jacobi iteration on field u
# returns rms residual
function iteration_2DPoisson!(u, f, h, c, alpha)
    nx, ny = size(u)

    # Do one Jacobi iteration
    res = @zeros(nx, ny)  # TODO preallocate this
    @parallel residual_2DPoisson!(u, f, h, c, res)

    # update r_rms
    r_rms = sqrt(sum(res.^2) / (nx * ny))

    # update u
    C = alpha * (h^2 / (4.0 + c * h^2))
    @parallel update_u!(res, C, u)

    @assert typeof(r_rms) == Float64
    return r_rms
end

@views function iteration_2DPoisson_gs!(u, f, h, c, alpha)

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

end

@parallel_indices (ix, iy) function restrict!(fine, coarse)
    nx, ny = size(fine)
    if (ix % 2 == 1 && iy % 2 == 1) && (3 <= ix <= nx-2 && 3 <= iy <= ny-2)
        ix_c, iy_c = ((ix-1)÷2)+1, ((iy-1)÷2)+1
        coarse[ix_c, iy_c] = fine[ix, iy]
    end
    return nothing
end

@views function restrict_wrapper!(fine, coarse, apply_BCs)
    # initalize coarse to zero (+ Dirichlet(0) BCs)
    coarse[:] .= 0.0

    @parallel restrict!(fine, coarse)

    # apply Neumann BCs for temperature T
    if apply_BCs
        apply_boundary_conditions_neumann!(coarse)
    end
end

@parallel_indices (ix, iy) function prolongate!(coarse, fine)
    nx, ny = size(fine)
    a2 = 1.0 / 2.0
    a4 = 1.0 / 4.0
    if (ix % 2 == 1 && iy % 2 == 1) && (3 <= ix <= nx-2 && 3 <= iy <= ny-2)
        ix_c, iy_c = ((ix-1)÷2)+1, ((iy-1)÷2)+1
        fine[ix, iy] = coarse[ix_c, iy_c]
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

@views function prolongate_wrapper!(coarse, fine, apply_BCs)
    # initialize fine to zero (+ Dirichlet(0) BCs)
    fine[:] .= 0.0

    @parallel prolongate!(coarse, fine)

    # apply Neumann BCs for temperature T
    if apply_BCs
        apply_boundary_conditions_neumann!(fine)
    end

end

@views function prolongate_wrapper_serial!(coarse, fine, apply_BCs)
    a2, a4 = 1.0 / 2.0, 1.0 / 4.0

    nx = size(fine, 1)
    ny = size(fine, 2)

    # initialize fine to zero (+ Dirichlet(0) BCs)
    fine .= 0.0

    jc = 2
    for j = 3:2: ny-2
        ic = 2
        for i = 3:2: nx-2
            fine[i, j] = coarse[ic, jc]
            fine[i+1, j] = fine[i+1, j] + a2 * coarse[ic, jc]
            fine[i-1, j] = fine[i-1, j] + a2 * coarse[ic, jc]
            fine[i, j+1] = fine[i, j+1] + a2 * coarse[ic, jc]
            fine[i, j-1] = fine[i, j-1] + a2 * coarse[ic, jc]
            fine[i+1, j+1] = fine[i+1, j+1] + a4 * coarse[ic, jc]
            fine[i+1, j-1] = fine[i+1, j-1] + a4 * coarse[ic, jc]
            fine[i-1, j+1] = fine[i-1, j+1] + a4 * coarse[ic, jc]
            fine[i-1, j-1] = fine[i-1, j-1] + a4 * coarse[ic, jc]
            ic = ic + 1
        end
        jc = jc + 1
    end

    # apply Neumann BCs for temperature T
    if (apply_BCs)
        apply_neumann_boundary_conditions(fine)
    end

end

@views function  restrict_wrapper_serial!(fine, coarse, apply_BCs)
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
