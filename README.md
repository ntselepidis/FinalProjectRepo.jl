# Final Project Repo

[![Build Status](https://github.com/eth-vaw-glaciology/FinalProjectRepo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/eth-vaw-glaciology/FinalProjectRepo.jl/actions/workflows/CI.yml?query=branch%3Amain)

This repository contains our final project for the course [**Solving PDEs in parallel on GPUs with Julia**](https://eth-vaw-glaciology.github.io/course-101-0250-00/) given in the Fall semester of 2021 at ETH Zurich.
The project consists of two main parts.
In part 1, we implement a 3D multi-XPU diffusion solver computing the steady-state solution of a diffusive process for given physical timesteps using the pseudo-transient acceleration (using the so-called "dual-time" method).
In part 2, we implement a 2D XPU Navier-Stokes solver that is based on the stream-vorticity formulation and allows both explicit and semi-implicit timestepping.
In every timestep of the Navier-Stokes simulation, we solve the required linear systems using a geometric Multigrid method that leads to a highly efficient matrix-free implementation.
We dive deeper into details for each part, and present various related analyses and results in the dedicated docs for [**part 1**](/docs/part1.md) and [**part 2**](/docs/part2.md).
