using ReferenceTests, Test
include("../scripts-part2/part2.jl")

opt = SimIn_t()
opt.nx, opt.ny = 257, 65
opt.W_init_strategy = W_from_file
sim_results = navier_stokes_2D(; opt=opt, verbose=false, do_vis=false, testmode=true)

comp_size(lhs, rhs) = size(lhs) == size(rhs)

function comp_val(nx, ny; atol=1e-8)
    comp_val_inner(lhs::Matrix{Float64}, rhs::Matrix{Float64}) =
        all(abs.(lhs[2:nx-1, 2:ny-1] - rhs[2:nx-1, 2:ny-1]) .< atol)
    return comp_val_inner
end

function my_test_reference(binfile, arr; comp=(==))
    bin_data = open(binfile, "r") do io
        return load(io)
    end
    @test comp(bin_data, arr)
end

# Add unit and reference tests
@testset "Test against FORTRAN reference implementation: array sizes." begin
    my_test_reference("test/reftest-files/fortran/T.bin", sim_results.T, comp=comp_size)
    my_test_reference("test/reftest-files/fortran/W.bin", sim_results.W, comp=comp_size)
    my_test_reference("test/reftest-files/fortran/S.bin", sim_results.S, comp=comp_size)
end

@testset "Test against FORTRAN reference implementation: array values." begin
    atol = 1e-8
    my_test_reference("test/reftest-files/fortran/T.bin", sim_results.T, comp=comp_val(opt.nx, opt.ny, atol=atol))
    my_test_reference("test/reftest-files/fortran/W.bin", sim_results.W, comp=comp_val(opt.nx, opt.ny, atol=atol))
    my_test_reference("test/reftest-files/fortran/S.bin", sim_results.S, comp=comp_val(opt.nx, opt.ny, atol=atol))
end
@ParallelStencil.reset_parallel_stencil()
