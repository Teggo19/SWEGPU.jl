include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie
 
n = 25

edges, cells = SWEGPU.make_structured_mesh(n, n)

n_refined = 200
edges_refined, cells_refined = SWEGPU.make_structured_mesh(n_refined, n_refined)

u0 = x -> [-0.125*(sign((x[1]-0.5)^2+ (x[2]-0.5)^2 -0.03) - 1)+ 1, 0.0, 0.0]

initial = hcat([u0(cell.centroid) for cell in cells]...)'

viz_initial = SWEGPU.visualize_height(initial, cells, edges)
viz(viz_initial, alpa=0.5, color=:green)



initial_refined = SWEGPU.refine_structured_grid(initial, n, n_refined)

viz_initial_refined = SWEGPU.visualize_height(initial_refined, cells_refined, edges_refined)
viz(viz_initial_refined, showsegments=true)

n_list = [8, 16, 32, 64]
T = 0.05
SWEGPU.convergence_test(n_list, T, u0, "Convergence test")

res = SWEGPU.SWE_solver(cells, edges, T, initial)

viz_res = SWEGPU.visualize_height(res, cells, edges)
viz(viz_res, alpa=0.5, color=:blue)