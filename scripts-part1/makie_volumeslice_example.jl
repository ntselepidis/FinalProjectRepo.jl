using GLMakie
# using WGLMakie

fig = Figure()
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

x = LinRange(0, π, 50);
y = LinRange(0, 2π, 100);
z = LinRange(0, 3π, 150);
t = LinRange(0.1, 1., 50);

lsgrid = labelslidergrid!(
  fig,
  ["yz plane - x axis", "xz plane - y axis", "xy plane - z axis", "time t"],
  [1:length(x), 1:length(y), 1:length(z), 1:length(t)]
);
fig[2, 1] = lsgrid.layout;

vol4 = [cos(X)*sin(Y)*sin(Z)*T for X ∈ x, Y ∈ y, Z ∈ z, T ∈ t];
vol = @view vol4[:, :, :, 1];
plt = volumeslices!(ax, x, y, z, vol)

# connect sliders to `volumeslices` update methods
sl_yz, sl_xz, sl_xy, sl_t = lsgrid.sliders

on(sl_yz.value) do v; plt[:update_yz][](v) end
on(sl_xz.value) do v; plt[:update_xz][](v) end
on(sl_xy.value) do v; plt[:update_xy][](v) end
on(sl_t.value) do v;
  plt[:volume][] = @view vol4[:, :, :, v];
  plt[:update_xy][](sl_xy.value[]);
  plt[:update_xz][](sl_xz.value[]);
  plt[:update_yz][](sl_yz.value[]);
end

set_close_to!(sl_yz, .5length(x))
set_close_to!(sl_xz, .5length(y))
set_close_to!(sl_xy, .5length(z))

# cam3d!(ax.scene, projectiontype=Makie.Orthographic)

fig

try
  while true
    for t_ ∈ eachindex(t)
      sl_t.value[] = t_
      sleep(0.1)
    end
    for t_ ∈ reverse(eachindex(t))
      set_close_to!(sl_t, t_)
      sleep(0.1)
    end
  end
catch InterruptException
  nothing
end
