include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie

supersimple_path = "/home/trygve/Master/Kode/Grids/supersimple.obj"


vs, fs = SWEGPU.read_obj(supersimple_path)

edges, cells = SWEGPU.generate_mesh(vs, fs)

initial_condition = zeros(Float64, length(cells), 3)
initial_condition[1, 1] = 1.0
initial_condition[2, 1] = 1.0


viz_height_initial = SWEGPU.visualize_height(initial_condition, cells, edges)
viz(viz_height_initial, color=:green)


T = 10.0
res = SWEGPU.SWE_solver(cells, edges, T, initial_condition)



viz_height = SWEGPU.visualize_height(res, cells, edges)
viz!(viz_height, color=:blue)