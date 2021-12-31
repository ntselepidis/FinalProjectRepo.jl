using DataFrames
using CSV
using PGFPlotsX

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\pgfplotsset{compat=1.7}")

@pgf common_opts = {
  title_style={align="center", font=raw"\large"}, 
  xlabel=raw"num MPI ranks",
  xlabel_style={align="center", font=raw"\Large",},
  xtick="{1, 2, 4, 8}",
  xmin=0.7,
  ylabel=raw"$\mathit{Perf} = \frac{\mathit{Work}}{t_{\rm elapsed}}\ \left[GFlop/s\right]$",
  ylabel_style={align="center", font=raw"\Large"},
  log_ticks_with_fixed_point,
  grid="major",
  legend_pos="outer north east",
}

@pgf bar_opts = {
  common_opts...,
  ymin=0,
  ybar,
  bar_width=1.2,
  area_legend,
}

for device in ["gpu", "cpu"]
  local ax1, ax2, ax3, ax4, gp, fig, df, df_use_shmem, df_wo_shmem
  df_device = DataFrame(CSV.File("benchmark-results/bench_diffusion_scaling_$(device).csv"))

  push!(bar_opts, :title => string(uppercase(device), raw": Strong scaling experiment \\ $nx=ny=nz=2^7$, $\mathit{ttot}=2s$ \\ Note: $\mathit{Total\ Work} = const.$"),)
  push!(common_opts, :xmax => (device == "gpu" ? 5.5 : 10.5 ))
  push!(bar_opts, :xmax => (device == "gpu" ? 5.5 : 10.5 ))

  df = filter(:strong_scaling => ==(true), df_device)
  df_use_shmem = filter(:use_shared_memory => ==(true), df)
  df_wo_shmem = filter(:use_shared_memory => !=(true), df)
  ax1 = SemiLogXAxis(bar_opts,
             PlotInc(Table(:x => df_use_shmem.n_mpi_ranks, :y => df_use_shmem.Performance/1e9)),
             PlotInc(Table(:x => df_wo_shmem.n_mpi_ranks, :y => df_wo_shmem.Performance/1e9)),
             Legend(["shared memory", raw"hide communication"]),
            )

  ax2 = SemiLogXAxis(@pgf({bar_opts..., title="", ylabel=raw"Throughput $\left[GB/s\right]$"}),
             PlotInc(Table(:x => df_use_shmem.n_mpi_ranks, :y => df_use_shmem.Throughput/1e9)),
             PlotInc(Table(:x => df_wo_shmem.n_mpi_ranks, :y => df_wo_shmem.Throughput/1e9)),
             Legend(["shared memory", raw"hide communication"]),
            )

  df = filter(:strong_scaling => ==(false), df_device)
  df_use_shmem = filter(:use_shared_memory => ==(true), df)
  df_wo_shmem = filter(:use_shared_memory => !=(true), df)
  ax3 = LogLogAxis(@pgf({common_opts..., title=string(uppercase(device), raw": Weak scaling experiment \\ $nx=ny=nz=2^7$, $\mathit{ttot}=2s$ \\ Note: $\mathit{Total\ Work} \propto \mathit{num\ ranks}$")}),
             PlotInc(Table(:x => df_use_shmem.n_mpi_ranks, :y => df_use_shmem.Performance/1e9)),
             PlotInc(Table(:x => df_wo_shmem.n_mpi_ranks, :y => df_wo_shmem.Performance/1e9)),
             # PlotInc(Table(:x => df_wo_shmem.n_mpi_ranks, :y => df_wo_shmem.Work/1e12)),
             Legend(["shared memory", raw"hide communication"]),
            )

  ax4 = SemiLogXAxis(@pgf({common_opts..., title="", ylabel=raw"Throughput $\left[GB/s\right]$", ymin=0.0}),
             PlotInc(Table(:x => df_use_shmem.n_mpi_ranks, :y => df_use_shmem.Throughput/1e9)),
             PlotInc(Table(:x => df_wo_shmem.n_mpi_ranks, :y => df_wo_shmem.Throughput/1e9)),
             Legend(["shared memory", raw"hide communication"]),
            )

  gp = GroupPlot(@pgf({group_style={group_size="2 by 2", horizontal_sep="7cm", vertical_sep="1.8cm"}}))
  push!(gp, ax1, ax3, ax2, ax4)
  fig = TikzPicture(gp)

  pgfsave("figs/diffusion_scaling_experiments_$(device).png", fig)
  pgfsave("figs/diffusion_scaling_experiments_$(device).pdf", fig)
end
