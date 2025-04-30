include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie

supersimple_path = "/home/trygve/Master/Kode/Grids/supersimple.obj"
path = "/home/trygve/Master/Kode/Grids/terrain.obj"

vs, fs = SWEGPU.read_obj(path)

edges, cells = SWEGPU.generate_mesh(vs, fs, Float64, Int64)

initial_condition = zeros(Float64, length(cells), 3)
initial_condition[1, 1] = 1.0
initial_condition[2, 1] = 1.0

viz_ground = SWEGPU.visualize_cells(cells)
viz_edges = SWEGPU.visualize_edges(edges)
viz(viz_ground)
viz!(viz_edges[1], color=:black)
viz_height_initial = SWEGPU.visualize_height(initial_condition, cells, edges)
viz(viz_height_initial, color=:green)


T = 10.0
res = SWEGPU.SWE_solver(cells, edges, T, initial_condition)



viz_height = SWEGPU.visualize_height(res, cells, edges)
viz!(viz_height, color=:blue)