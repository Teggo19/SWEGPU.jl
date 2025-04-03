include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie
using BenchmarkTools

#obj_path = "/home/trygve/Master/Kode/Grids/terrain.obj"
#test_path = "/home/trygve/Master/Kode/Grids/simple.obj"

#vs, fs = SWEGPU.read_obj(test_path)
#vs, fs = SWEGPU.read_obj(supersimple_path)

#edges, cells = SWEGPU.generate_mesh(vs, fs)

#initial_condition = zeros(Float64, length(cells), 3)
#initial_condition[1:4, 1] .= 2.0
#initial_condition[5:8, 1] .= 1.5


#viz_bottom, _ = SWEGPU.visualize_cells(cells)

#viz_height_initial = SWEGPU.visualize_height(initial_condition, cells, edges)
#viz(viz_height_initial, color=:green)

#viz_water = SWEGPU.visualize_water(initial_condition, cells)

#viz(viz_water, color=:blue, alpha=0.8, showsegments=true)
#viz!(viz_bottom, color=:grey)
#viz(viz_height, color=:blue, alpha=0.5)


#T = 1
#res = SWEGPU.SWE_solver(cells, edges, T, initial_condition)



#viz_height = SWEGPU.visualize_height(res, cells, edges)
#viz!(viz_height, color=:blue)
n_x, n_y = 100, 100
include("../mesh_generating/structured.jl")
vs, fs = make_mesh(n_x, n_y);
edges, cells = SWEGPU.generate_mesh(vs, fs);
#viz_bottom, _ = SWEGPU.visualize_cells(cells)
#viz(viz_bottom, showsegments=true)

u0 = x -> [exp.(-((x[1]-0.5)^2+(x[2]-0.5)^2)/0.0001)+ 1, 0.0, 0.0]

#u0 = x -> [1.0, 0.0, 0.0]

x = SWEGPU.cell_centres(cells)

initial = hcat([u0(x[i, :]) for i in 1:size(x)[1]]...)'

viz_height_initial = SWEGPU.visualize_height(initial, cells, edges)

viz(viz_height_initial)

T = 0.02

res = SWEGPU.SWE_solver(cells, edges, T, initial)

viz_height_res = SWEGPU.visualize_height(res, cells, edges)
viz(viz_height_res)#, color=:green)

include("convert_to_cartesian.jl")
convert_to_cartesian(res, n_x, n_y, "t02x100x100")