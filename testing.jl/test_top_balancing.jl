include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie
using CUDA

n = 100
#n2 = 20
#n3 = 40


top_func = x -> 10*x[1]^2 #topography function
edges, cells= SWEGPU.make_structured_mesh_with_topography(n, n, Float32, Int64, top_func);

#edges2, cells2 = SWEGPU.make_structured_mesh(n2, n2, Float32, Int64);
#edges3, cells3 = SWEGPU.make_structured_mesh(n3, n3, Float32, Int64);
#viz_bottom = SWEGPU.visualize_cells(cells)
#viz(viz_bottom, color=1:length(cells))


#u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
u0 = x -> [11.f0, 0.f0, 0.f0]
#u0 = x -> [0.5f0 + 0.2f0*exp((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2), 0.f0, 0.f0]

quad = SWEGPU.quadrature(u0, cells)




res = SWEGPU.SWE_solver(cells, edges, 0.1f0, quad; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false)

viz_res = SWEGPU.visualize_height(res, cells, edges)
viz(viz_res)

maximum([res[i, 1] + cells[i].centroid[3] for i in 1:length(cells)])
minimum([res[i, 1] + cells[i].centroid[3] for i in 1:length(cells)])