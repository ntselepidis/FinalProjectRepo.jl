using ParallelStencil
using LinearAlgebra

include("part2_utils.jl")

@parallel_indices (ix, iy) function matrix_free_matvec_prod!(T, hx, hy, c, dT2)
    if (1 < ix < size(T, 1) && 1 < iy < size(T, 2))
        dT2[ix, iy] = (( T[ix+1, iy] - 2*T[ix, iy] + T[ix-1, iy] ) / hx^2 +
                       ( T[ix, iy+1] - 2*T[ix, iy] + T[ix, iy-1] ) / hy^2) - c * T[ix, iy]
    end
    return nothing
end

@parallel_indices (ix, iy) function matrix_free_matvec_prod_shmem!(T_, hx, hy, c, dT2)
    tx = @threadIdx().x + 1
    ty = @threadIdx().y + 1
    T = @sharedMem(eltype(T_), (@blockDim().x+2, @blockDim().y+2))
    if (1 <= ix <= size(T_, 1) && 1 <= iy <= size(T_, 2))
        T[tx, ty] = T_[ix, iy]
    end
    @sync_threads()
    if (1 < ix < size(T_, 1) && 1 < iy < size(T_, 2))
        if (@threadIdx().x == 1)             T[tx-1, ty] = T_[ix-1, iy] end
        if (@threadIdx().x == @blockDim().x) T[tx+1, ty] = T_[ix+1, iy] end
        if (@threadIdx().y == 1)             T[tx, ty-1] = T_[ix, iy-1] end
        if (@threadIdx().y == @blockDim().y) T[tx, ty+1] = T_[ix, iy+1] end
        @sync_threads()
        dT2[ix, iy] = (( T[tx+1, ty] - 2*T[tx, ty] + T[tx-1, ty] ) / hx^2 +
                       ( T[tx, ty+1] - 2*T[tx, ty] + T[tx, ty-1] ) / hy^2) - c * T[tx, ty]
    end
    return nothing
end


function matrix_free_matvec_prod_wrapper!(p, hx, hy, c, p_hat; execution_policy=parallel_shmem)
    nx, ny = size(p)
    if execution_policy == parallel_shmem
        threads = (32, 8)
        blocks = ((nx, ny) .- 1) .÷ (threads) .+ 1
        shmem = prod(threads.+2) * sizeof(Float64)
        @parallel blocks threads shmem=shmem matrix_free_matvec_prod_shmem!(p, hx, hy, c, p_hat)
    elseif execution_policy == parallel
        @parallel matrix_free_matvec_prod!(p, hx, hy, c, p_hat)
    elseif execution_policy == serial
        error()
    else
        error()
    end
    @synchronize()
end

"""
Krylov method solving (Δ - c)*x = b
"""
function cg!(x_in, b, hx, hy, c, tol, Nmax; execution_policy=parallel_shmem, verbose=false)
    nx, ny = size(b)
    normb  = norm(b)
    tolb   = tol * normb
    r      = copy(b)
    p      = copy(r)
    p_hat  = copy(r)
    x      = @zeros(nx, ny)
    normr  = Inf
    rho    = sum(r.*r)

    for i = 1:Nmax
        # p_hat .= A*p
        matrix_free_matvec_prod_wrapper!(p, hx, hy, c, p_hat, execution_policy=execution_policy)
        alpha = rho / sum(p.*p_hat)
        x .+= alpha * p
        r .-= alpha * p_hat
        normr = norm(r)
        if verbose
            println("$(i): $(normr/normb)")
        end
        if (normr < tolb)
            if verbose
                println("CG converged in $(i) iterations :)")
            end
            break
        end
        rho_old = rho
        rho = sum(r.*r)
        beta = rho / rho_old
        p .= r + beta * p
    end

    x_in .= x

    return sqrt(sum(r.^2) / (nx * ny))
end
