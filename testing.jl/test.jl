include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie
using CUDA

n = 1024

@time edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64);
#edges2, cells2 = SWEGPU.make_structured_mesh(2048, 2048, Float32, Int64);


#n_refined = 200
#edges_refined, cells_refined = SWEGPU.make_structured_mesh(n_refined, n_refined)

#u0 = x -> [-0.125f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.03f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
u0 = x -> [-0.f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.03f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
#u0 = x -> [0.5f0 + 1f0*x[1], 0.f0, 0.f0]


initial = hcat([u0(cell.centroid) for cell in cells]...)';

viz_height_initial = SWEGPU.visualize_height(initial, cells, edges)
#viz(viz_height_initial)

#viz_recon = SWEGPU.visualize_reconstruction(res, edges, cells)
#viz(viz_recon, color=1:length(cells))

res = SWEGPU.SWE_solver(cells, edges, 0.1f0, initial; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true)
CUDA.@profile res2 = SWEGPU.SWE_solver(cells, edges, 0.1f0, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=true)
N_list = [16, 32, 64, 128, 256, 512, 1024, 2048]

viz_recon = SWEGPU.visualize_reconstruction(initial, edges, cells)
viz(viz_recon, color=1:length(cells))
@time n_list, diffs, f, baseline_n, ord_conv = SWEGPU.convergence_test(N_list, 0.2f0, u0, "Convergence test", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=0)

@time n_list2, diffs2, f2, baseline_n, ord_conv2 = SWEGPU.convergence_test(N_list, 0.2f0, u0, "Convergence test", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=1)

