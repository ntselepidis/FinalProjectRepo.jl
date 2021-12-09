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

function dist_norm_L2(Rh, comm_cart)
    sq_residual = [sum(Rh .^ 2)]  # this is local
    MPI.Allreduce!(sq_residual, +, comm_cart)
    return sqrt(sq_residual[1])
end
