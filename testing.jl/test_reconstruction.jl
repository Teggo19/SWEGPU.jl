include("../src/SWEGPU.jl")
using .SWEGPU
using GLMakie
using Meshes


T = 0.0f0
u0_cont = x -> [0.5f0*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02)+0.02, 0.f0, 0.f0]

n_baseline = 2048

edges_baseline, cells_baseline = SWEGPU.make_structured_mesh(n_baseline, n_baseline, Float64, Int64);
initial_cont = SWEGPU.quadrature(u0_cont, cells_baseline);

cont_baseline_swegpu = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_cont; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, return_runtime=false, CFL=0.5f0, limiter=1);

#SWEGPU.radial_plot(cont_baseline_swegpu, cells_baseline)
#recon_gradients = SWEGPU.make_reconstructions(cells_baseline, edges_baseline, cont_baseline_swegpu; limiter=1)

#viz_res = SWEGPU.visualize_height(cont_baseline_swegpu, cells_baseline, edges_baseline)
#viz(viz_res)


#abs_recon_gradients = sqrt.(recon_gradients[:, 1, 1].^2 .+ recon_gradients[:, 1, 2].^2)

#SWEGPU.radial_plot(abs_recon_gradients, cells_baseline)

#save("tmp/SWEGPU/circular_dam_break_n=$(n_baseline)_t=$(T).jld", "res", baseline_swegpu)

N_list = [32, 64, 128, 256, 512]

n_list_test, diffs_test, f, baseline_n, ord_conv_test = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T, u0_cont, "edge limiter"; backend="CUDA", time_stepper=1, limiter=0)

_, diffs, f2, _, _ = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T, u0_cont, "minmod limiter"; 
                                                                     backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=1, userecon=true, spaceType=Float64)

_, diffs2, f3, _, _ = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T, u0_cont, "0 limiter"; 
                                                                     backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=2, userecon=true)

                                                                     

#save("tmp/recon_test/edge_limiter.png", f)
#save("tmp/recon_test/minmod_limiter.png", f2)
#save("tmp/recon_test/0_limiter_test.png", f3)
