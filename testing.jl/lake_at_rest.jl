include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie
using CUDA

n = 1024

es, cs = SWEGPU.make_structured_mesh(n, n, Float32, Int64, alternative=false);

u0 = x -> [(1.f0), 0.f0, 0.f0]
initial = SWEGPU.quadrature(u0, cs);


T = 1.0f0

res = SWEGPU.SWE_solver(cs, es, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=0, CFL=0.5f0, limiter=2);

maximum(abs.(res .- initial))

top_func = x -> 0.5*exp(-50.f0*(x[1]-0.5f0)^2 - 50.f0*(x[2]-0.5f0)^2)

u02 = x -> [1.f0 - top_func(x), 0.f0, 0.f0]

n2 = 1024
es2, cs2 = SWEGPU.make_structured_mesh_with_topography(n2, n2, Float32, Int64, top_func);


initial2 = SWEGPU.quadrature(u02, cs2);


res2 = SWEGPU.SWE_solver(cs2, es2, 1.f0, initial2; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=0, CFL=0.5f0, limiter=2);

maximum(abs.(res2 .- initial2))

SWEGPU.L1(res2, initial2)