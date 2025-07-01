include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie
using CUDA

n = 1024

es, cs = SWEGPU.make_structured_mesh(n, n, Float32, Int64, alternative=false);

#u0 = x -> [0.002*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02)+0.02, 0.f0, 0.f0]
#u0 = x -> [0, 0.05f0*((x[1]-0.5f0)^2 + (x[2]-0.5f0)^2), 0]
#u0 = x -> [x[1] + 0.5, 0f0, 0f0]
u0 = x -> [(-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0), 0.f0, 0.f0]
initial = SWEGPU.quadrature(u0, cs);

T = 0.022f0

#res = SWEGPU.SWE_solver(cs, es, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=0, return_runtime=true);
#res2 = SWEGPU.SWE_solver(cs, es, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=1, return_runtime=true);

res_new = SWEGPU.SWE_solver(cs, es, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=0, return_runtime=true);
res2_new = SWEGPU.SWE_solver(cs, es, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=1, return_runtime=true);

#runtime before optimization
# limiter = 0: 4.3s
# limiter = 1: 3.9s

# runtime after optimization of flux computation
# limiter = 0: 4.3s
# limiter = 1: 3.9s

# runtime after optimization of cell update
