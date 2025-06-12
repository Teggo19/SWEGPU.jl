include("../src/SWEGPU.jl")
using .SWEGPU
using BenchmarkTools
using Meshes
using GLMakie
using CUDA
using Cthulhu

n = 1024

edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64);

u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
# u0 = x -> [11.f0, 0.f0, 0.f0]
#u0 = x -> [0.5f0 + 0.2f0*exp((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2), 0.f0, 0.f0]

quad = SWEGPU.quadrature(u0, cells)

T = 0.10f0/n

res = SWEGPU.SWE_solver(cells, edges, T, quad; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false)
