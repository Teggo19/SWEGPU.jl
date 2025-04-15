include("../src/SWEGPU.jl")
using .SWEGPU
#using BenchmarkTools
using Meshes
using GLMakie
 
n = 100

edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64)

#n_refined = 200
#edges_refined, cells_refined = SWEGPU.make_structured_mesh(n_refined, n_refined)

u0 = x -> [-0.125f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.03f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
#u0 = x -> [-0.f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.03f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

initial = hcat([u0(cell.centroid) for cell in cells]...)'

viz_initial = SWEGPU.visualize_height(initial, cells, edges)
viz(viz_initial, alpa=0.5, color=:green)



#initial_refined = SWEGPU.refine_structured_grid(initial, n, n_refined)

#viz_initial_refined = SWEGPU.visualize_height(initial_refined, cells_refined, edges_refined)
#viz(viz_initial_refined, showsegments=true)

#n_list = [8, 16, 32, 64]
T = 0.02f0
#SWEGPU.convergence_test(n_list, T, u0, "Convergence test")

res = SWEGPU.SWE_solver(cells, edges, T, initial)

viz_res = SWEGPU.visualize_height(old_res, cells, edges)
viz(viz_res, alpa=0.5, color=:blue)

SWEGPU.radial_plot(res, cells)

using JLD

old_res = JLD.load("tmp/CUDA_test/old_n=100.jld")
old_res = old_res["cpu"]

gpu_res = JLD.load("tmp/CUDA_test/new_n=100.jld", "gpu")
SWEGPU.radial_plot(res, cells)
SWEGPU.radial_plot(new_res, cells)
SWEGPU.radial_plots([old_res, res], cells, ["old", "new"])

