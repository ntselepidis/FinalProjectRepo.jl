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

function linear_interpolate_3D(H, dx)
    dy = dz = dx

    ix, iy, iz = Int.((LOCATION_OF_INTEREST.÷ dx) .+ 1)
    
    # manual trilinear interpolation
    # see https://en.wikipedia.org/wiki/Trilinear_interpolation#Alternative_algorithm
    x0, x1 = ix*dx+dx/2, (ix+1)*dx+dx/2
    y0, y1 = iy*dy+dy/2, (iy+1)*dy+dy/2
    z0, z1 = iz*dz+dz/2, (iz+1)*dz+dz/2

    cvec = [H[ix, iy, iz], H[ix+1, iy, iz], H[ix, iy+1, iz], H[ix+1, iy+1, iz], H[ix, iy, iz+1], H[ix+1, iy, iz+1], H[ix, iy+1, iz+1], H[ix+1, iy+1, iz+1]]

    xvec = repeat([x0;x1], 4)
    yvec = repeat([y0; y0; y1; y1], 2)
    zvec = repeat(repeat([z0], 4), 2)

    M = hcat(ones(8), xvec, yvec, zvec, xvec.*yvec, xvec.*zvec, yvec.*zvec, xvec.*yvec.*zvec)

    interp_value = try
        avec = M \ cvec
        x, y, z = LOCATION_OF_INTEREST
        xvec_ = [1 x y z x*y x*z y*z x*z*z]
        interp_value = xvec_ * avec
        interp_value
    catch e
        interp_value = H[ix, iy, iz]
        interp_value
    end
end
