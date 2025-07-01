include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie
using CUDA

n = 2048

es, cs = SWEGPU.make_structured_mesh(n, n, Float32, Int64, alternative=false);
viz_cells = SWEGPU.visualize_cells(cs)
viz(viz_cells, showsegments=true)



#u0 = x -> [0.002*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02)+0.02, 0.f0, 0.f0]
#u0 = x -> [0, 0.05f0*((x[1]-0.5f0)^2 + (x[2]-0.5f0)^2), 0]
#u0 = x -> [x[1] + 0.5, 0f0, 0f0]
u0 = x -> [(-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0), 0.f0, 0.f0]
initial = SWEGPU.quadrature(u0, cs);

viz_recon = SWEGPU.visualize_reconstruction(initial, es, cs; limiter=1)
#viz(viz_recon, showsegments=true, alpha=0.5f0)

viz_initial = SWEGPU.visualize_height(initial, cs, es)
viz(viz_initial, showsegments=true)
viz!(viz_initial[90], color=:red)
viz!(viz_initial[91], color=:blue)

SWEGPU.radial_plots([initial], cs, ["Initial condition"], "test")

T = 0.02f0

res = SWEGPU.SWE_solver(cs, es, T, initial; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=1, CFL=0.5f0, limiter=1);
res2 = SWEGPU.SWE_solver(cs, es, T, initial; backend="cpu", bc=SWEGPU.neumannBC, reconstruction=0, CFL=0.5f0, limiter=1);

SWEGPU.radial_plots([res], cs, ["SWEGPU recon"], "test")

viz_res = SWEGPU.visualize_height(res, cs, es)
viz(viz_res)