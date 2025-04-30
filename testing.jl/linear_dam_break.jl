include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie
using BenchmarkTools
using StaticArrays

include("../comparing_with_SinFVM/file_helper.jl")
include("../comparing_with_SinFVM/converting.jl")

n = 50

n_x, n_y = n, n
edges, cells = SWEGPU.make_structured_mesh(n_x, n_y, Float32, Int64);

function initial_f(x)
    return [1.0 + 0.25*(sign(1-(x[2]+x[1]))+1), 0.0, 0.0]
end

# initial_cartesian = read_from_file("SinFVM/linear_dam_break_n=$(n)_t=0")
# initial = cartesian_to_triangular(initial_cartesian)

initial = Matrix{Float64}(undef, n_x*n_y*2, 3)

for i in 1:2*n_x*n_y
    initial[i, :] = initial_f(cells[i].centroid)
end

viz_height_initial = SWEGPU.visualize_height(initial, cells, edges)

viz(viz_height_initial)

T = 0.2f0
SWEGPU.dynamic_visualization(edges, cells, initial, T, 0.005f0; bc=SWEGPU.wallBC)

@time res = SWEGPU.SWE_solver(cells, edges, T, initial; bc=SWEGPU.wallBC, backend="cpu")

viz_height_res = SWEGPU.visualize_height(res, cells, edges)
viz(viz_height_res, alpha=1)#, color=:green)


result_SinFVM_cartesian = read_from_file("SinFVM/linear_dam_break_n=$(n)_t=0.1")
result_SinFVM = cartesian_to_triangular(result_SinFVM_cartesian)

viz_res_SinFVM = SWEGPU.visualize_height(result_SinFVM, cells, edges)

viz!(viz_res_SinFVM, color=:green, alpha = 0.7)

SWEGPU.radial_plot(res, cells)
SWEGPU.radial_plot(res - result_SinFVM, cells)

SWEGPU.radial_plots([res, result_SinFVM], cells, ["triangular", "SinFVM"])

N_list = [8, 16, 32, 64, 128, 256]
T = 0.2f0

@time N_list, baseline, diffs, f = SWEGPU.convergence_test(N_list, T, initial_f, "Convergence test linear dam break"; backend="CUDA", bc=SWEGPU.wallBC)

using JLD
JLD.save("tmp/convergence_test/linear_dam_break.jld", "N_list", N_list, "baseline", baseline, "diffs", diffs)
