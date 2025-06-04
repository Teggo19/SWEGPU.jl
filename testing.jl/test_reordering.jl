include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie
using BenchmarkTools

file_path = "../Grids/terrain.obj"
vertices, faces = SWEGPU.read_obj(file_path)

edges, cells = SWEGPU.generate_mesh(vertices, faces, Float32, Int64)

viz_cells = SWEGPU.visualize_cells(cells)
viz(viz_cells, color=1:length(cells))

new_vertices, new_faces = SWEGPU.reorder_triangular_grid(vertices, faces)

new_edges, new_cells = SWEGPU.generate_mesh(new_vertices, new_faces, Float32, Int64)

viz_new_cells = SWEGPU.visualize_cells(new_cells)
viz(viz_new_cells, color=1:length(new_cells))

x_min = minimum(new_vertices[1, :])
x_max = maximum(new_vertices[1, :])
y_min = minimum(new_vertices[2, :])
y_max = maximum(new_vertices[2, :])
z_min = minimum(new_vertices[3, :])
z_mak = maximum(new_vertices[3, :])
x_range = x_max - x_min
y_range = y_max - y_min
range = min(x_range, y_range)

u0 = x -> [-0.25f0*(sign((x[1]-(x_min + x_max)*0.5f0)^2+ (x[2]-(y_min + y_max)*0.5f0)^2 -0.01f0*range) - 1.f0)+ 60.f0, 0.f0, 0.f0]

initial = SWEGPU.quadrature(u0, cells)
new_initial = SWEGPU.quadrature(u0, new_cells)

viz_initial = SWEGPU.visualize_height(initial, cells, edges)
viz(viz_initial, color=:blue, alpha=0.5)
viz!(viz_new_cells)

@time res = SWEGPU.SWE_solver(cells, edges, 0.02f0, initial; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=0, return_runtime=false)
@time new_res = SWEGPU.SWE_solver(new_cells, new_edges, 0.02f0, new_initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, return_runtime=false)