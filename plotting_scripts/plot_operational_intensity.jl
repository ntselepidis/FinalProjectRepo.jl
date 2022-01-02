using DataFrames
using CSV
using PGFPlotsX

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\pgfplotsset{compat=1.7}")
df_gpu = DataFrame(CSV.File("benchmark-results/bench_diffusion_scaling_gpu.csv"))

op_intensity_shmem = df_gpu[1, :Intensity]
op_intensity_hide_comm = df_gpu[2, :Intensity]

ax = Axis(@pgf({ylabel=raw"[Flop/Bytes]",
                ylabel_style={align="center", font=raw"\Large"},
                title="Operational Intensity",
                title_style={font=raw"\large"},
                xmajorticks="false",
                ybar, ymin=0, bar_width=0.5, area_legend, legend_pos="outer north east"}),
          PlotInc(Table(:x => 0, :y => [op_intensity_shmem])),
          PlotInc(Table(:x => 0, :y => [op_intensity_hide_comm])),
          Legend(["shared memory", "hide communication"])
)

pgfsave("figs/diffusion_scaling_experiments_operational_intensity.png", ax)
pgfsave("figs/diffusion_scaling_experiments_operational_intensity.pdf", ax)
