include("../src/SWEGPU.jl")
using .SWEGPU
using GLMakie
using Meshes

vertices, faces = SWEGPU.read_obj("../Grids/test_fine.obj")

x_min = minimum(vertices[1, :])
x_max = maximum(vertices[1, :])
y_min = minimum(vertices[2, :])
y_max = maximum(vertices[2, :])
z_min = minimum(vertices[3, :])
z_max = maximum(vertices[3, :])
x_range = x_max - x_min
y_range = y_max - y_min
range = min(x_range, y_range)
vertices = vertices .- [x_min, y_min, z_min]


#edges, cells = SWEGPU.generate_mesh(vertices, faces, Float32, Int64);
#SWEGPU.save_structures(cells, edges, "meshes/test_fine_original")

edges, cells = SWEGPU.load_structures("meshes/test_fine_original")

#viz_cells = SWEGPU.visualize_cells(cells)
#viz(viz_cells, color=1:length(cells))
#viz(viz_cells[end], color=:red)
#println("Reordering cells...")
#new_vertices, new_faces = SWEGPU.reorder_triangular_grid(vertices, faces)
#println("Constructing new mesh...")
#new_edges, new_cells = SWEGPU.generate_mesh(new_vertices, new_faces, Float32, Int64)
#SWEGPU.save_structures(new_cells, new_edges, "meshes/test_fine_reordered")
new_edges, new_cells = SWEGPU.load_structures("meshes/test_fine_reordered")

viz_new_cells = SWEGPU.visualize_cells(new_cells)
viz(viz_new_cells, showsegments=true)

#u0 = x -> [-25f0*(sign((x[1]-(x_min + x_max)*0.5f0)^2+ (x[2]-(y_min + y_max)*0.5f0)^2 -0.01f0*range^2) - 1.f0)+ 60.f0, 0.f0, 0.f0]
u0 = x -> [z_max + 10f0, 0.f0, 0.f0]
initial = SWEGPU.quadrature(u0, cells)
new_initial = SWEGPU.quadrature(u0, new_cells)

viz_initial = SWEGPU.visualize_height(initial, cells, edges)
viz!(viz_initial, color=:blue, alpha=0.5)
viz!(viz_new_cells, alpho=0.5)

res, _ = SWEGPU.SWE_solver(cells, edges, 0.1f0, initial; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true);
new_res, _ = SWEGPU.SWE_solver(new_cells, new_edges, 0.1f0, new_initial; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true)

viz_res = SWEGPU.visualize_height(res, cells, edges)
viz(viz_res, showsegments=true)

scuffed_cells = []
for i in 1:length(new_cells)
    if sum(abs.(new_cells[i].grad)) > 7.5
        push!(scuffed_cells, viz_new_cells[i])
    end
end
viz!(scuffed_cells, color=:red)

viz_cells_2d = SWEGPU.visualize_cells_2D(new_cells)
viz(viz_cells_2d, color=1:length(cells))

# save the visualization
save("tmp/test_fine_visualization.png", viz)