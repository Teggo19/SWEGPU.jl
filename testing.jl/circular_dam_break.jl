include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie
using BenchmarkTools
using StaticArrays

include("../comparing_with_SinFVM/file_helper.jl")

n = 100

initial_cartesian = read_from_file("SinFVM/circular_dam_break_n=$(n)_t=0")

include("../comparing_with_SinFVM/converting.jl")
initial = cartesian_to_triangular(initial_cartesian)

n_x, n_y = n, n
include("../mesh_generating/structured.jl")
vs, fs = make_mesh(n_x, n_y);
edges, cells = SWEGPU.generate_mesh(vs, fs);


viz_height_initial = SWEGPU.visualize_height(initial, cells, edges)

viz(viz_height_initial)

T = 0.02

res = SWEGPU.SWE_solver(cells, edges, T, initial)

viz_height_res = SWEGPU.visualize_height(res, cells, edges)
viz(viz_height_res, alpha=1)#, color=:green)

result_SinFVM_cartesian = read_from_file("SinFVM/circular_dam_break_n=$(n)_t=0.02")
result_SinFVM = cartesian_to_triangular(result_SinFVM_cartesian)

viz_res_SinFVM = SWEGPU.visualize_height(result_SinFVM, cells, edges)

viz!(viz_res_SinFVM, color=:green, alpha = 0.7)

SWEGPU.radial_plot(res, cells)
SWEGPU.radial_plot(result_SinFVM, cells)

SWEGPU.radial_plots([res, result_SinFVM], cells, ["triangular", "SinFVM"])