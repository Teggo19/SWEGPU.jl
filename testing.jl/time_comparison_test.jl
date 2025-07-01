include("../src/SWEGPU.jl")
using .SWEGPU
using GLMakie

# circular dam break
u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

N_list = [32, 64, 128, 256, 512, 1024, 2048]

T = 0.022f0
#f4, cpu2, gpu2, n_timesteps = SWEGPU.compare_runtime([], T, u0, "CPU vs CUDA, test"; time_stepper=1, spaceType=Float64)


#f3, cpu1, gpu1 = SWEGPU.compare_runtime(N_list, T, u0, "CPU vs CUDA, first order"; bc=SWEGPU.neumannBC, recon=0)

f4, cpu2, gpu2, n_timesteps = SWEGPU.compare_runtime(N_list, T, u0, "CPU vs CUDA, second order"; bc=SWEGPU.neumannBC, time_stepper=1, limiter=1, spaceType=Float64)


for (i, N) in enumerate(N_list)
    println("\$2^{$(i+4)}\$ & $(round(gpu1[i], digits=5)) & $(round(cpu1[i], digits=5)) & $(round(gpu2[i], digits=5)) & $(round(cpu2[i], digits=5)) \\\\")
end


# Float 64 results
SinFVM_cpu_times = [0.008877992630004883, 0.02196192741394043, 0.07739996910095215, 0.3589489459991455, 3.186267852783203, 23.243678092956543, 159.50010299682617]
SinFVM_gpu_times = [0.008831977844238281, 0.03697800636291504, 0.11842203140258789, 0.4963998794555664, 2.40071177482605, 15.880714178085327, 112.74244379997253]
SinFVM_n_timesteps = [12, 24, 47, 95, 190, 381, 762]

SWEGPU_cpu_times = [0.002755880355834961, 0.008144855499267578, 0.05140399932861328, 0.4024059772491455, 3.961521863937378, 32.62599492073059, 236.1259150505066]
SWEGPU_gpu_times = [0.0044171810150146484, 0.008272886276245117, 0.04282212257385254, 0.20502185821533203, 1.2795579433441162, 9.64706802368164, 73.85864901542664]
SWEGPU_n_timesteps = [10, 20, 40, 81, 162, 325, 651]

f = Figure(resolution=(1200, 600))
ax = Axis(f[1, 1], xlabel="N", ylabel="Time/timestep (s)", title="Time Comparison: SinFVM vs SWEGPU", xscale=log10, yscale=log10)
lines!(ax, N_list, SinFVM_cpu_times./ SinFVM_n_timesteps, label="SinFVM CPU", color=:blue, linestyle=:dash)
scatter!(ax, N_list, SinFVM_cpu_times./ SinFVM_n_timesteps, color=:blue, markersize=10)
lines!(ax, N_list, SinFVM_gpu_times./ SinFVM_n_timesteps, label="SinFVM GPU", color=:green)
scatter!(ax, N_list, SinFVM_gpu_times./ SinFVM_n_timesteps, color=:green, markersize=10)
lines!(ax, N_list, SWEGPU_cpu_times./ SWEGPU_n_timesteps, label="SWEGPU CPU", color=:red, linestyle=:dash)
scatter!(ax, N_list, SWEGPU_cpu_times./ SWEGPU_n_timesteps, color=:red, markersize=10)
lines!(ax, N_list, SWEGPU_gpu_times./ SWEGPU_n_timesteps, label="SWEGPU GPU", color=:orange)
scatter!(ax, N_list, SWEGPU_gpu_times./ SWEGPU_n_timesteps, color=:orange, markersize=10)

f[1, 2] = Legend(f, ax, "Plots")
save("tmp/time_comp/time_comparison.png", f)

for (i, N) in enumerate(N_list)
    println("\$2^{$(i+4)}\$ & $(round(SinFVM_cpu_times[i]/SinFVM_n_timesteps[i], digits=5)) & $(round(SinFVM_gpu_times[i]/SinFVM_n_timesteps[i], digits=5)) & $(round(SWEGPU_cpu_times[i]/SWEGPU_n_timesteps[i], digits=5))& $(round(SWEGPU_gpu_times[i]/SWEGPU_n_timesteps[i], digits=5)) \\\\")
end