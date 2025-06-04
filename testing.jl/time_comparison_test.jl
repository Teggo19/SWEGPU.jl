include("../src/SWEGPU.jl")
using .SWEGPU

# circular dam break
u0 = x -> [-0.125f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.03f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

N_list = [64, 128, 256, 512, 1024]

T = 0.2f0

f3, _, _ = SWEGPU.compare_runtime(N_list, T, u0, "CPU vs CUDA, first order", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=0)

f4, _, _ = SWEGPU.compare_runtime(N_list, T, u0, "CPU vs CUDA, second order", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=1)