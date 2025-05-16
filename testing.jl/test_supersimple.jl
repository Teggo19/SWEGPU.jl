include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie

supersimple_path = "/home/trygve/Master/Kode/Grids/supersimple.obj"
path = "/home/trygve/Master/Kode/Grids/terrain.obj"

vs, fs = SWEGPU.read_obj(path)

edges, cells = SWEGPU.generate_mesh(vs, fs, Float64, Int64)

u0 = x -> [-0.125f0*(sign((x[1]-262400)^2+ (x[2]-6650400)^2 -10000) - 1.f0)+ 1.f0, 0.f0, 0.f0]

initial = hcat([u0(cell.centroid) for cell in cells]...)';
#initial_condition[1, 1] = 1.0
#initial_condition[2, 1] = 1.0

viz_ground = SWEGPU.visualize_cells(cells)
viz_edges = SWEGPU.visualize_edges(edges)
viz(viz_ground)
viz!(viz_edges[1], color=:black)
viz_height_initial = SWEGPU.visualize_height(initial, cells, edges)
viz(viz_height_initial)


T = 10.0
res = SWEGPU.SWE_solver(cells, edges, T, initial_condition)



viz_height = SWEGPU.visualize_height(res, cells, edges)
viz!(viz_height, color=:blue)