include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie
using CUDA

n = 100
#n2 = 20
#n3 = 40


top_func = x -> 0.0*x[1] #topography function
edges, cells= SWEGPU.make_structured_mesh_with_topography(n, n, Float32, Int64, top_func);

#edges2, cells2 = SWEGPU.make_structured_mesh(n2, n2, Float32, Int64);
#edges3, cells3 = SWEGPU.make_structured_mesh(n3, n3, Float32, Int64);
#viz_bottom = SWEGPU.visualize_cells(cells)
#viz(viz_bottom, color=1:length(cells))


#u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
#u0 = x -> [1.f0, 0.f0, 0.f0]
u0 = x -> [0.5f0 + 0.2f0*exp((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2), 0.f0, 0.f0]

quad = SWEGPU.quadrature(u0, cells)
quad_reconstructions = SWEGPU.make_reconstructions(cells, edges, quad)
res_reconstructions = SWEGPU.make_reconstructions(cells, edges, res)

sum(SWEGPU.L1_quadrature_recon(quad, res, quad_reconstructions, quad_reconstructions, cells))

initial = hcat([u0(cell.centroid) - [cell.centroid[3], 0.f0, 0.f0] for cell in cells]...)';

viz_quad = SWEGPU.visualize_height(quad, cells, edges)
viz_initial = SWEGPU.visualize_height(initial, cells, edges)
viz!(viz_initial)
viz!(viz_quad, color=:red, alpha=0.5)

viz_cells = SWEGPU.visualize_cells(cells)
viz(viz_cells)




res = SWEGPU.SWE_solver(cells, edges, 0.02f0, quad; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false)

viz_res = SWEGPU.visualize_height(res, cells, edges)
viz(viz_res)
viz_recon = SWEGPU.visualize_reconstruction(res, edges, cells)
viz!(viz_recon)#, color=1:length(cells))

recon = SWEGPU.make_reconstructions(cells, edges, res)

ref2 = SWEGPU.refine_structured_grid(res, n, n2)
ref2_recon = SWEGPU.refine_structured_grid(res, n, n2, recon)

viz_ref2 = SWEGPU.visualize_height(ref2, cells2, edges2)
viz_ref2_recon = SWEGPU.visualize_height(ref2_recon, cells2, edges2)
viz!(viz_initial2, color=:green, alpha=0.8)
viz!(viz_initial2_recon, color=:red, alpha=0.5)

viz_diff = SWEGPU.visualize_height(ref2 .- ref2_recon, cells2, edges2)
viz(viz_diff)
viz_recon = SWEGPU.visualize_reconstruction(initial, edges, cells)
viz!(viz_recon, color=1:length(cells))