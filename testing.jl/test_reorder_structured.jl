include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie
using BenchmarkTools

n = 256
edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64);

edges2, cells2 = SWEGPU.make_structured_reordered_mesh(n, n, Float32, Int64);
edges3, cells3 = SWEGPU.make_shuffled_mesh(n, n, Float32, Int64);

viz_cells = SWEGPU.visualize_cells_2D(cells)
viz_cells2 = SWEGPU.visualize_cells_2D(cells2)
viz_cells3 = SWEGPU.visualize_cells_2D(cells3)
viz(viz_cells, color=1:length(cells))
viz(viz_cells2, color=1:length(cells2))
viz(viz_cells3, color=1:length(cells3))

# save the visualization
save("tmp/figures/structured_mesh.png", viz(viz_cells, color=1:length(cells)))
save("tmp/figures/structured_reordered_mesh.png", viz(viz_cells2, color=1:length(cells2)))
save("tmp/figures/structured_shuffled_mesh.png", viz(viz_cells3, color=1:length(cells3)))

u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

initial = SWEGPU.quadrature(u0, cells);
initial2 = SWEGPU.quadrature(u0, cells2);
initial3 = SWEGPU.quadrature(u0, cells3);

T = 0.22f0

res, t1 = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true, CFL=0.5f0);
res2, t2 = SWEGPU.SWE_solver(cells2, edges2, T, initial2; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true, CFL=0.5f0);
res3, t3 = SWEGPU.SWE_solver(cells3, edges3, T, initial3; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true, CFL=0.5f0);


res, s1 = SWEGPU.SWE_solver(cells, edges, T, initial; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true, CFL=0.5f0);
res2, s2 = SWEGPU.SWE_solver(cells2, edges2, T, initial2; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true, CFL=0.5f0);
res3, s3 = SWEGPU.SWE_solver(cells3, edges3, T, initial3; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true, CFL=0.5f0);

