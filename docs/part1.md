# Part 1: 3D multi-XPU 3D Diffusion Solver

## Introduction & Method

In part 1, we implement a 3D multi-XPU diffusion solver computing the steady-state solution of a diffusive process (see equation (1)) for given physical timesteps using an implicit approach leveraging pseudo-transient acceleration (using the so-called ["dual-time"](https://eth-vaw-glaciology.github.io/course-101-0250-00/lecture4/#implicit_solutions) method).

<p align="center">
  <img src="./figs-part1/math/1_diffusion_3d.png" />
</p>

For additional reading related to this method, we refer the reader [here](https://eth-vaw-glaciology.github.io/course-101-0250-00/lecture4/#steady-state_and_implicit_iterative_solutions).

### Domain Shape, Initial and Boundary Conditions

We solve the diffusion equation on a cubic domain.

As initial condition, we define a Gaussian distribution of H centered in the domain's center with amplitude 2 and standard deviation 1.

Moreover, we enforce Dirichlet boundary condition H = 0 on all 6 faces.

### Implementation

In our implementation we use [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) for the (multi-)XPU implementation.

We implemented the solver both using [array](https://github.com/ntselepidis/FinalProjectRepo.jl/blob/main/scripts-part1/part1_array_programming.jl) and using [kernel](https://github.com/ntselepidis/FinalProjectRepo.jl/blob/main/scripts-part1/part1_kernel_programming.jl) programming.

However, we focus on the analysis of the solver and report results only for our kernel programming implementation.

For the kernel programming approach we implemented the solver such that it can use `shared memory` or `hide_communication`.

Ideally we should combine both approaches, however, this is not possible with the current implementation of [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl), so in our experiments we consider them mutually exclusive.

## Results

### Hardware

We conducted all numerical experiments on a single compute node of octopus.

Such a node has 2 `Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz` (6 cores each) and 4 `NVIDIA GeForce GTX TITAN X` GPUs.

An overview of the architecture of the node is given below:

```
ntselepidis@node15 ➜  ~ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    mlx4_0  mlx4_1  CPU Affinity    NUMA Affinity
GPU0     X      PHB     SYS     SYS     PHB     SYS     0-5     0
GPU1    PHB      X      SYS     SYS     PHB     SYS     0-5     0
GPU2    SYS     SYS      X      PHB     SYS     PHB     6-11    1
GPU3    SYS     SYS     PHB      X      SYS     PHB     6-11    1
mlx4_0  PHB     PHB     SYS     SYS      X      SYS
mlx4_1  SYS     SYS     PHB     PHB     SYS      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

```

We note that for an `NVIDIA GeForce GTX TITAN X` the theoretical performance for double-precision arithmetic is 209.1 GFLOPS, and the memory bandwidth is 336.6 GB/s.
For an `Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz` the memory bandwidth is 59 GB/s.

### Example Simulation

Below we show an animation of the 3D solution of the diffusive process.

<p align="center">
  <img src="./figs-part1/diffusion3d.gif" width="750" height="750" />
</p>

### Assessing Performance

#### Operational Intensity

We instrumented our code and computed the operational intensity for the two approaches.
We summarize our finding in the following plot.

<p align="center">
  <img src="./figs-part1/diffusion_scaling_experiments_operational_intensity.png" />
</p>

Using shared memory, allows the reuse of "intermediate" data and thus reducing the number of loads and stores.
Therefore, given that the number of flops for the two approaches is the same, the approach that uses shared memory leads to higher operational intensity.

#### Strong and Weak Scaling

In this section we present the results of our multi-XPU strong and weak scaling experiments.
When running on the **CPU**, we map **one MPI rank per core** (and do not use multi-threading).
When running on **GPU**s, we map **one MPI rank per GPU**.
In our strong scaling experiments we increase the total number of MPI ranks while keeping the overall work constant.
In our weak scaling experiments we increase the total number of MPI ranks, and also proportionally increase the overall work, so that the work per MPI rank is constant.
We run all experiments using both our kernel programming approaches, i.e.:

- using `shared memory` (without `hide_communication`),
- using `hide_communication` (without `shared memory`),

and report performance [GFlop/s], throughput [GB/s], and time [s].

**Notes**:
- It should be mentioned, that the operational intensity of the two approaches is different because using shared memory substantially reduces the bytes moved (see previous section).
Therefore, the throughput plots do not directly show the difference in runtime.
This is not the case in our performance plots, since the GFlops in both our approaches are the same.
- We expect that using `shared memory` through ParallelStencil.jl does not have an impact on the CPU since this is purely a GPU feature.

Below we show the part of our benchmark [code](https://github.com/ntselepidis/FinalProjectRepo.jl/blob/main/scripts-part1/part1_scaling_experiments.jl) that computes the local problem sizes in our strong and weak scaling experiments along with some reasoning about our choice in the comments.
```
# We need 7 buffers for the solver
# memory(nx) = nx^3 * sizeof(Float64) * 7
# => with nx = ny = nz = 2^9 we have memory(nx) ≈ 7.5GB => Fits on Titan X (12 GB)
# After experimenting with 2^9 we found it very hard to converge.
# Empirically, nx = ny = nz = 2^7 gives a good trade-off between convergence speed and precision.
nx, ny, nz =
    if strong_scaling
        dims_dict = Dict(
                         1 => (1, 1, 1),
                         2 => (2, 1, 1),
                         4 => (2, 2, 1),
                         8 => (2, 2, 2),
                        )
        2^7 .÷ dims_dict[MPI.Comm_size(MPI.COMM_WORLD)]  # global 2^7 gets divided to determine local size
    else
        2^7 .* (1, 1, 1)  # local 2^7 (assigned to every MPI rank)
    end
```

We also note, that in order to ensure the consistency of our experiments we scale the physical dimensions `lx, ly, lz` when we conduct weak scaling experiments, to keep the workload constant per MPI rank.
We achieve this with the [code](https://github.com/ntselepidis/FinalProjectRepo.jl/blob/main/scripts-part1/part1_kernel_programming.jl) shown below:
```
# Allow scaling the physical problem dimensions with the number of grid points.
# This is useful for weak scaling experiments where we want to keep the work per worker (i.e. MPI rank) fixed.
# In particular, note that usually increasing nx -> 2*nx => dx -> dx/2 => dτ = dτ/4  => !nonlinear!
# Therefore, by scaling the physical size together with (nx, ny, nz) we can keep (dx, dy, dz) constant and therefore keep the work per worker constant.
if scale_physical_size
    lx, ly, lz = dims .* (10., 10., 10.)
else
    lx, ly, lz = 10.0, 10.0, 10.0
end
```

**Strong and Weak Scaling on the CPU**

<p align="center">
  <img src="./figs-part1/diffusion_scaling_experiments_cpu.png" />
</p>

**Observations**
- Both the approach that uses `shared memory` and the approach using `hide_communication` exhibit great strong as well as weak scaling on the CPU.
- Interestingly, the fastest variant is the one that does not use `hide_communication`.
- In terms of memory throughput, the approach that uses `hide_communication` is the one that achieves higher throughput, but as mentioned earlier this is because in the instrumentation we compute that more bytes are being moved, and not because of the total runtime.
- In all cases it seems that we are compute bound.

**Strong and Weak Scaling on the GPU**

<p align="center">
  <img src="./figs-part1/diffusion_scaling_experiments_gpu.png" />
</p>

**Observations**
- Both the approach that uses `shared memory` and the approach using `hide_communication` exhibit great strong as well as weak scaling on the GPU.
- In the **strong scaling** plots we see a performance increase when moving from 1 GPU to 2 GPUs and a small performance drop when moving from 2 GPUs to 4 GPUs.
This is probably because when moving from 1 GPU to 2 GPUs we transition from a compute bound regime to a memory bound regime, whereas when moving from 2 GPUs to 4 GPUs we transition from a memory bound regime to a communication bound regime.
This can also be observed from the following:
    - when running on 1 GPU using shared memory does not really impact the runtime (since we are compute bound)
    - when running on 2 GPUs using shared memory leads to smaller runtime (since we are memory bound)
    - when running on 4 GPUs using hide communication leads to smaller runtime (since we are communication bound)
- In the **weak scaling** plots we see that increasing the problem size, while keeping the work fixed per GPU, leads to almost constant runtime (while increasing performance and throughput).
- Moreover, in this case using `hide_communication` seems to gradually bring more and more benefit while increasing problem size (and proportionally the number of GPUs used).

#### Work-precision diagrams

In the following figures, we plot the solution behaviour (i.e. the evolution of a value from the quantity we are diffusing for a specific location) as a function of the solver's tolerance (left), and as function of the numerical grid resolution (right).

<p align="center">
  <img src="./figs-part1/work_precision_diagrams.png" />
</p>

**Observations**
- Increasing the solver's tolerance makes sense up to tol = 1e-6 since after this point the solution seems to have converged to a certain value.
- Increasing the numerical grid resolution makes the solution at a certain point to gradually converge to a certain value.
