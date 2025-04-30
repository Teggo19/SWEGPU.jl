include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie

n = 100

edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64);

#n_refined = 200
#edges_refined, cells_refined = SWEGPU.make_structured_mesh(n_refined, n_refined)

u0 = x -> [-0.125f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.03f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
#u0 = x -> [-0.f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.03f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]



initial = hcat([u0(cell.centroid) for cell in cells]...)';

SWEGPU.dynamic_visualization(edges, cells, initial, 0.1f0, 0.005f0)


viz_initial = SWEGPU.visualize_height(initial, cells, edges)
viz(viz_initial)



#initial_refined = SWEGPU.refine_structured_grid(initial, n, n_refined)

#viz_initial_refined = SWEGPU.visualize_height(initial_refined, cells_refined, edges_refined)
#viz(viz_initial_refined, showsegments=true)

n_list = [8, 16, 32, 64, 128, 256]
T = 0.1f0

@time N_list, baseline, diffs, f = SWEGPU.convergence_test(n_list, T, u0, "Convergence test"; backend="cpu", bc=SWEGPU.wallBC)

#using JLD
#JLD.save("tmp/convergence_test/circular_dam_break.jld", "N_list", N_list, "baseline", baseline, "diffs", diffs)


@time res_cpu = SWEGPU.SWE_solver(cells, edges, T, initial)
@time res_gpu = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA")

viz_res = SWEGPU.visualize_height(res_gpu, cells, edges)
viz(viz_res, alpa=0.5, color=:blue)

SWEGPU.radial_plot(res, cells)

using JLD

old_res = JLD.load("tmp/CUDA_test/old_n=100.jld")
old_res = old_res["cpu"]

gpu_res = JLD.load("tmp/CUDA_test/new_n=100.jld", "gpu")
SWEGPU.radial_plot(res, cells)
SWEGPU.radial_plot(new_res, cells)
SWEGPU.radial_plots([res_cpu, res_gpu], cells, ["cpu", "gpu"])

