using FileIO
using SparseArrays

@enum ExecutionPolicy_t begin
    serial          # uses Gauss-Seidel smoother and coarse solver, and no ParallelStencil code at all
    parallel        # uses Jacobi smoother and coarse solver, and ParallelStencil code
    parallel_shmem  # uses Jacobi smoother and coarse solver, and ParallelStencil code + shared memory for residual_2DPoisson
end

""" Loads double-precision matrices in binary format, leading with nx and ny. """
function load(f)
    nx = Array{Int32}(undef, 1, 1)
    ny = Array{Int32}(undef, 1, 1)
    read!(f, nx)
    read!(f, ny)
    sim_results = Array{Float64}(undef, nx[1, 1], ny[1, 1])
    read!(f, sim_results)
    return sim_results
end

""" Applies Dirichlet and Neumann boundary conditions on all 4 edges of the rectangular domain. """
@views function apply_boundary_conditions!(T)
    apply_boundary_conditions_dirichlet!(T)
    apply_boundary_conditions_neumann!(T)
end

""" Applies Dirichlet boundary conditions on bottom and top edges of the rectangular domain. """
@views function apply_boundary_conditions_dirichlet!(T)
    nx, ny = size(T)
    T[:, 1]  .= 1.0  # High temperature on the bottom edge
    T[:, ny] .= 0.0  # Low temperature on the top edge
end

""" Applies Neumann boundary conditions on left and right edges of the rectangular domain. """
@views function apply_boundary_conditions_neumann!(T)
    nx, ny = size(T)
    T[1, :]  .= T[2, :]
    T[nx, :] .= T[nx-1, :]
end

""" Creates sparse coefficient matrix as defined by 5-point stencil. """
@views function stencil_5pt(nx, ny)
    Δx = spdiagm(-1 => 1 * ones(nx - 1), 0 => -2 * ones(nx), 1 => 1 * ones(nx-1))
    Δy = spdiagm(-1 => 1 * ones(ny - 1), 0 => -2 * ones(ny), 1 => 1 * ones(ny-1))
    Ix = spdiagm(0 => ones(nx))
    Iy = spdiagm(0 => ones(ny))
    Δ = kron(Δy, Ix) + kron(Iy, Δx)
    return Δ
end
