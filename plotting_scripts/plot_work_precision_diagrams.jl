using DataFrames
using CSV
using PGFPlotsX

df = DataFrame(CSV.File("benchmark-results/error_vs_tolerance_experiment_results.csv"))

# dual-time method or pseudo-transient acceleration method
@pgf ax1 = LogLogAxis(
                {xlabel=raw"convergence tolerance \\ of inner timestep",
                 title=raw"Diffusion value after Δt = 2 seconds \\ at $(x, y, z) = (4.5, 4.5, 4.5)$ \\ versus solver tolerance",
                 ylabel=raw"solution value at $(x, y, z)$",
                 title_style={align="center"},
                 xlabel_style={align="center"},
                 x_dir="reverse",
                 grid="major"},
     Plot(Coordinates(zip(df.tol, df.val)))
)


df = DataFrame(CSV.File("benchmark-results/error_vs_grid_size_experiment_results.csv"))

# dual-time method or pseudo-transient acceleration method
@pgf ax2 = SemiLogXAxis(
                {xlabel=raw"nx = ny = nz",
                 title=raw"Diffusion value after Δt = 2 seconds \\ at $(x, y, z) = (4.5, 4.5, 4.5)$ \\ versus grid resolution",
                 ylabel=raw"solution value at $(x, y, z)$",
                 title_style={align="center"},
                 xlabel_style={align="center"},
                 log_basis_x=2,
                 grid="major"},
     Plot(Coordinates(zip(df.nx, df.val)))
)

@pgf gp = GroupPlot({group_style={group_size="2 by 1", horizontal_sep="2cm"}})
push!(gp, ax1, ax2)
fig = TikzPicture(gp)

pgfsave("figs/work_precision_diagrams.png", fig)
pgfsave("figs/work_precision_diagrams.pdf", fig)
