#!/bin/bash
set -e

module load julia cuda/11.4

# We use CUDA-aware MPI
# You can check: `julia -e "using MPI; @assert MPI.has_cuda()"`
julia -E 'using MPI; "MPI.has_cuda() = $(MPI.has_cuda())"'
julia -e 'using MPI; @assert MPI.has_cuda()'
export IGG_CUDAAWARE_MPI=1

mkdir -p benchmark-results

for device in "gpu" "cpu"; do
    for scaling_experiment in "strong" "weak"; do

        FILENAME="benchmark-results/bench_diffusion_${scaling_experiment}_scaling_${device}.csv"

        if [[ -f ${FILENAME} ]]; then
            echo "Already created ${FILENAME}. Moving to next benchmark."
        else
            for nprocs in 1 2 4; do
                ~/.julia/bin/mpiexecjl -np ${nprocs} julia --project -O3 --check-bounds=no scripts-part1/part1_scaling_experiments.jl ${device} ${scaling_experiment}
            done
            # For CPU we can go up to 8 MPI processes
            if [[ ${device} == "cpu" ]]; then
                nprocs=8
                ~/.julia/bin/mpiexecjl -np ${nprocs} julia --project -O3 --check-bounds=no scripts-part1/part1_scaling_experiments.jl ${device} ${scaling_experiment}
            fi
            echo "Generated ${FILENAME} ..."
        fi

    done
done
