using FileIO

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
