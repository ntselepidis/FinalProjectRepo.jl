#!/bin/bash
set -e

module load julia cuda/11.4

# We use CUDA-aware MPI
# You can check: `julia -e "using MPI; @assert MPI.has_cuda()"`
julia -E 'using MPI; "MPI.has_cuda() = $(MPI.has_cuda())"'
julia -e 'using MPI; @assert MPI.has_cuda()'
export IGG_CUDAAWARE_MPI=1

mkdir -p benchmark-results

# Scaling experiments (weak and strong)
for device in "gpu" "cpu"; do
    FILENAME="benchmark-results/bench_diffusion_scaling_${device}.csv"

    if [[ -f ${FILENAME} ]]; then
        echo "Already created ${FILENAME}. Moving to next benchmark."
    else
        for nprocs in 1 2 4; do
            ~/.julia/bin/mpiexecjl -np ${nprocs} julia --project -O3 --check-bounds=no scripts-part1/part1_scaling_experiments.jl ${device}
        done
        # For CPU we can go up to 8 MPI processes
        if [[ ${device} == "cpu" ]]; then
            nprocs=8
            ~/.julia/bin/mpiexecjl -np ${nprocs} julia --project -O3 --check-bounds=no scripts-part1/part1_scaling_experiments.jl ${device}
        fi
        echo "Generated ${FILENAME} ..."
    fi
done

# Investigate error vs grid size
FILENAME="benchmark-results/error_vs_grid_size_experiment_results.csv"
if [[ -f ${FILENAME} ]]; then
    echo "Already created ${FILENAME}. Moving to next benchmark."
else
    julia --project -O3 --check-bounds=no scripts-part1/part1_error_vs_grid_size_experiments.jl gpu
fi

# Investigate error vs solver threshold
FILENAME="benchmark-results/error_vs_tolerance_experiment_results.csv"
if [[ -f ${FILENAME} ]]; then
    echo "Already created ${FILENAME}. Moving to next benchmark."
else
    julia --project -O3 --check-bounds=no scripts-part1/part1_error_vs_tolerance_experiments.jl gpu
fi
