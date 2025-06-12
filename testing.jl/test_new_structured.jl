include("../src/SWEGPU.jl")
using .SWEGPU
include("../comparing_with_SinFVM/file_helper.jl")
include("../comparing_with_SinFVM/converting.jl")
using GLMakie
using Meshes

T = 0.022f0
n_baseline = 512
baseline_quad_norecon = read_from_file("circular_dam_break_n=$(n_baseline)_t=$(T)_norecon")
baseline_norecon = cartesian_to_triangular(baseline_quad_norecon)

baseline_quad = read_from_file("circular_dam_break_n=$(n_baseline)_t=$(T)")
baseline = cartesian_to_triangular(baseline_quad)

# circular dam break
u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
#u0 = x -> [-0.25f0*(sign((x[1]-0.4888f0)^2+ (x[2]-0.4888f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

n = 512

edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64);

initial = SWEGPU.quadrature(u0, cells);

#viz_initial = SWEGPU.visualize_height(initial, cells, edges)
#viz(viz_initial)

res_norecon = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, return_runtime=false);
res = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false);
#viz_res = SWEGPU.visualize_height(res, cells, edges)
#viz(viz_res)

SWEGPU.radial_plot(res, cells)

SWEGPU.radial_plots([res, res_norecon], cells, ["SWEGPU recon", "SWEGPU norecon"])
SWEGPU.radial_plots([res_norecon, baseline_norecon], cells, ["SWEGPU norecon", "SinFVM norecon"])
SWEGPU.radial_plots([res, res_norecon, baseline], cells, ["SWEGPU recon", "SWEGPU norecon", "SinFVM recon"])
SWEGPU.radial_plots([res, baseline], cells, ["SWEGPU recon", "SinFVM recon"])
SWEGPU.radial_plots([baseline_norecon, baseline], cells, ["SinFVM norecon", "SinFVM recon"])