include("../src/SWEGPU.jl")
using .SWEGPU
using GLMakie

# circular dam break
u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

N_list = [32, 64, 128, 256, 512, 1024, 2048]

T = 0.022f0

f3, cpu1, gpu1 = SWEGPU.compare_runtime(N_list, T, u0, "CPU vs CUDA, first order"; bc=SWEGPU.neumannBC, recon=0)

f4, cpu2, gpu2 = SWEGPU.compare_runtime(N_list, T, u0, "CPU vs CUDA, second order"; bc=SWEGPU.neumannBC, recon=1)

save("time_comparison_1st_order.png", f3)
save("time_comparison_2nd_order.png", f4)

for (i, N) in enumerate(N_list)
    println("\$2^{$(i+4)}\$ & $(round(gpu1[i], digits=5)) & $(round(cpu1[i], digits=5)) & $(round(gpu2[i], digits=5)) & $(round(cpu2[i], digits=5)) \\\\")
end
