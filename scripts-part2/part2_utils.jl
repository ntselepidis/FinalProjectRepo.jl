using FileIO
using SparseArrays

"Load double-precision matrices in binary format, leading with nx and ny."
function load(f)
    nx = Array{Int32}(undef, 1, 1)
    ny = Array{Int32}(undef, 1, 1)
    read!(f, nx)
    read!(f, ny)
    sim_results = Array{Float64}(undef, nx[1, 1], ny[1, 1])
    read!(f, sim_results)
    return sim_results
end

@views function apply_boundary_conditions!(T)
    apply_boundary_conditions_dirichlet!(T)
    apply_boundary_conditions_neumann!(T)
end

@views function apply_boundary_conditions_dirichlet!(T)
    nx, ny = size(T)
    T[:, 1]  .= 1.0
    T[:, ny] .= 0.0
end

@views function apply_boundary_conditions_neumann!(T)
    nx, ny = size(T)
    T[1, :]  .= T[2, :]
    T[nx, :] .= T[nx-1, :]
end

@views function stencil_5pt(nx, ny)
    Δx = spdiagm(-1 => 1 * ones(nx - 1), 0 => -2 * ones(nx), 1 => 1 * ones(nx-1))
    Δy = spdiagm(-1 => 1 * ones(ny - 1), 0 => -2 * ones(ny), 1 => 1 * ones(ny-1))
    Ix = spdiagm(0 => ones(nx))
    Iy = spdiagm(0 => ones(ny))
    Δ = kron(Δy, Ix) + kron(Iy, Δx)
    return Δ
end
