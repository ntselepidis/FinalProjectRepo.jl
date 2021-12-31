using DataFrames
using CSV
using PGFPlotsX

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\pgfplotsset{compat=1.7}")
df = DataFrame(CSV.File("benchmark-results/part2_semi_implicit_vs_explicit_experiment_results.csv"))

beta(b) = df -> filter(:beta => ==(b), df)

@pgf common_opts = {
  title_style={align="center", font=raw"\large"}, 
  xlabel_style={align="center", font=raw"\Large",},
  ylabel=raw"$\mathit{Perf} = \frac{\mathit{Work}}{t_{\rm elapsed}}\ \left[GFlop/s\right]$",
  ylabel_style={align="center", font=raw"\Large"},
  # log_ticks_with_fixed_point,
  grid="major",
  legend_pos="outer north east",
}

ax = LogLogAxis(@pgf({
                      common_opts...,
                      legend_pos="south east", 
                      title=raw"Solver comparison \\ $nx=2^{11}, ny=2^9, \mathit{ttot}=0.005s$",
                      xlabel = raw"Prandtl number \\ $Pr = \frac{\mathit{momentum\ diffusivity}}{\mathit{thermal\ diffusivity}}$",
                      ylabel="Solving time [s]"}),
          PlotInc(Table(beta(0.0)(df).Pr, beta(0.0)(df).t_elapsed)),
          PlotInc(Table(beta(0.5)(df).Pr, beta(0.5)(df).t_elapsed)),
          PlotInc(Table(beta(1.0)(df).Pr, beta(1.0)(df).t_elapsed)),
          Legend([raw"explicit ($\beta=0$)", raw"semi-implicit ($\beta = 0.5$)", raw"implicit ($\beta = 1.0$)"])
)

pgfsave("figs/part2_semi_implicit_vs_explicit_experiment_results.png", ax)
pgfsave("figs/part2_semi_implicit_vs_explicit_experiment_results.pdf", ax)
