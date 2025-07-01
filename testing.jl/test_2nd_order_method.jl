include("../src/SWEGPU.jl")
using .SWEGPU
include("../comparing_with_SinFVM/file_helper.jl")
include("../comparing_with_SinFVM/converting.jl")
using GLMakie
using Meshes

#T = 0.022f0
T = 0.022f0
n_baseline = 512
baseline_quad_norecon = read_from_file("circular_dam_break_n=$(n_baseline)_t=$(T)_norecon")
baseline_norecon = cartesian_to_triangular(baseline_quad_norecon)

baseline_quad = read_from_file("circular_dam_break_n=$(n_baseline)_t=$(T)")
baseline = cartesian_to_triangular(baseline_quad)

# circular dam break
u0 = x -> [(-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.011f0) - 1.f0)+ 1.f0), 0.f0, 0.f0]
#u0 = x -> [-0.25f0*(sign((x[1]-0.4888f0)^2+ (x[2]-0.4888f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

n = 512

edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64; alternative=false);

#viz_cells = SWEGPU.visualize_cells(cells)
#viz(viz_cells, showsegments=true)

initial = SWEGPU.quadrature(u0, cells; order=1);

#viz_initial = SWEGPU.visualize_height(initial, cells, edges)
#viz(viz_initial, )


res_norecon = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, return_runtime=false, CFL=0.5f0, limiter=2);
res = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false, CFL=0.5f0);
res2 = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false, CFL=0.5f0, limiter=1);
res3 = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false, CFL=0.5f0, limiter=2);
#viz_res = SWEGPU.visualize_height(res3, cells, edges)
#viz(viz_res, showsegments=true)

#viz_recon = SWEGPU.visualize_reconstruction(res2, edges, cells; limiter=1)
#viz(viz_recon, showsegments=true)

#viz_height = SWEGPU.visualize_height(res, cells, edges)
#viz(viz_height, showsegments=true)

#SWEGPU.radial_plot(res_norecon, cells)

f = SWEGPU.radial_plots([res_norecon, res2 ], cells, ["norecon", "edge_limiter"], "test")
g = SWEGPU.radial_plots([res_norecon, res3 ], cells, ["norecon", "0 limiter"], "test2")

save("tmp/recon_test/f1.png", f)
save("tmp/recon_test/f2.png", g)
#save("circular_dam_break_radial_plot_comparison_1st_order.png", f)
#save("circular_dam_break_radial_plot_comparison_2nd_order.png", f2)