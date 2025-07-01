include("../src/SWEGPU.jl")
using .SWEGPU
include("../comparing_with_SinFVM/file_helper.jl")
include("../comparing_with_SinFVM/converting.jl")
using GLMakie

T = 0.022f0
T2 = 0.04f0
n_baseline = 2048
#baseline_quad = read_from_file("circular_dam_break_n=$(n_baseline)_t=$(T)")
#baseline_sinfvm = cartesian_to_triangular(baseline_quad)
# circular dam break
u0_circ = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
#u0_cont = x -> [0.2*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02) + 1.f0, 0.f0, 0.f0]
u0_cont = x -> [0.002*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02)+0.02, 0.f0, 0.f0]

edges_baseline, cells_baseline = SWEGPU.make_structured_mesh(n_baseline, n_baseline, Float64, Int64);
initial_circ = SWEGPU.quadrature(u0_circ, cells_baseline);
initial_cont = SWEGPU.quadrature(u0_cont, cells_baseline);

circ_baseline_swegpu = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_circ; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, return_runtime=false, CFL=0.5f0, limiter=1);
cont_baseline_swegpu = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T2, initial_cont; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, return_runtime=false, CFL=0.5f0, limiter=1);

#circ_baseline_swegpu_1st, time, n_timesteps = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_circ; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=0, return_runtime=true, CFL=0.5f0);
#cont_baseline_swegpu_1st = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T2, initial_cont; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=0, return_runtime=false, CFL=0.5f0);
#using JLD
#save("tmp/SWEGPU/circular_dam_break_n=$(n_baseline)_t=$(T).jld", "res", baseline_swegpu)

N_list = [32, 64, 128, 256, 512]

n_list, diffs, f, baseline_n, ord_conv = SWEGPU.convergence_test(circ_baseline_swegpu, n_baseline, N_list, T, u0_circ, "Convergence test first order solver circ"; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=0, CFL=0.5f0, limiter=2)

#n_list, diffs2, f2, baseline_n, ord_conv2 = SWEGPU.convergence_test(circ_baseline_swegpu, n_baseline, N_list, T, u0_circ, "Convergence test second order solver circ edge-limiter"; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, altmesh=false)
n_list, diffs3, f3, baseline_n, ord_conv3 = SWEGPU.convergence_test(circ_baseline_swegpu, n_baseline, N_list, T, u0_circ, "Convergence test second order solver circ minmod-limiter"; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=1)

n_list3, diffs4, g, baseline_n, ord_conv4 = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T2, u0_cont, "Convergence test first order solver cont"; 
                                                                    backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=2, spaceType=Float64)

#n_list4, diffs5, g2, baseline_n, ord_conv5 = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T2, u0_cont, "Convergence test second order solver cont edge-limiter"; 
#                                                                    backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=0)
n_list5, diffs6, g3, baseline_n, ord_conv6 = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T2, u0_cont, "Convergence test second order solver cont minmod-limiter";
                                                                    backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, limiter=1, spaceType=Float64)
                                                            

#save("tmp/convergence_test/convergence_test_circ_1st_order.png", f)
#save("tmp/convergence_test/convergence_test_circ_2nd_order.png", f3)
save("tmp/convergence_test/convergence_test_cont_1st_order.png", g)
save("tmp/convergence_test/convergence_test_cont_2nd_order.png", g3)
#save("tmp/convergence_test/circular_dam_break_radial_plot_comparison_2nd_order.png", f2)


for (i, N) in enumerate(N_list)
    if i == 1
        c = 0
        c2 = 0
        c3 = 0
        c4 = 0
    else
        c = log(2, diffs[i-1]/diffs[i])
        c2 = log(2, diffs3[i-1]/diffs3[i])
        c3 = log(2, diffs4[i-1]/diffs4[i])
        c4 = log(2, diffs6[i-1]/diffs6[i])
    end
    println("\$2^{$(i+4)}\$ & $(round(diffs[i], digits=5))& $(round(c, digits=5)) & $(round(diffs3[i], digits=5)) & $(round(c2, digits=5)) & $(round(diffs4[i], digits=9)) & $(round(c3, digits=5)) & $(round(diffs6[i], digits=9)) & $(round(c4, digits=5))\\\\")
end