using ParallelStencil
using LinearAlgebra

@parallel_indices (ix, iy) function matrix_free_matvec_prod!(T, hx, hy, c, dT2)
    if (1 < ix < size(T, 1) && 1 < iy < size(T, 2))
        dT2[ix, iy] = (( T[ix+1, iy] - 2*T[ix, iy] + T[ix-1, iy] ) / hx^2 +
                       ( T[ix, iy+1] - 2*T[ix, iy] + T[ix, iy-1] ) / hy^2) - c * T[ix, iy]
    end
    return nothing
end

"""
Krylov method solving (Δ - c)*x = b
"""
function cg!(x_in, b, hx, hy, c, tol, Nmax; verbose=false)
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
        @parallel matrix_free_matvec_prod!(p, hx, hy, c, p_hat)
        @synchronize()
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
