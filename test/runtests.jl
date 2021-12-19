using FinalProjectRepo
using Test, ReferenceTests, BSON

# make sure to turn off GPU usage, at least for Github Actions

include("part1.jl")
include("part2.jl")
include("multigrid.jl")
