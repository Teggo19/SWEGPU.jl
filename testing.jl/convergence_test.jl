include("../src/SWEGPU.jl")
using .SWEGPU
include("../comparing_with_SinFVM/file_helper.jl")
include("../comparing_with_SinFVM/converting.jl")

T = 0.022f0
T2 = 0.04f0
n_baseline = 2048
baseline_quad = read_from_file("circular_dam_break_n=$(n_baseline)_t=$(T)")
baseline_sinfvm = cartesian_to_triangular(baseline_quad)
# circular dam break
u0_circ = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
#u0_cont = x -> [0.2*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02) + 1.f0, 0.f0, 0.f0]
u0_cont = x -> [0.002*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02)+0.02, 0.f0, 0.f0]

edges, cells = SWEGPU.make_structured_mesh(n_baseline, n_baseline, Float32, Int64);
initial_circ = SWEGPU.quadrature(u0_circ, cells);
initial_cont = SWEGPU.quadrature(u0_cont, cells);

circ_baseline_swegpu = SWEGPU.SWE_solver(cells, edges, T, initial_circ; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false, CFL=0.5f0);
cont_baseline_swegpu = SWEGPU.SWE_solver(cells, edges, T2, initial_cont; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false, CFL=0.5f0);

circ_baseline_swegpu_1st = SWEGPU.SWE_solver(cells, edges, T, initial_circ; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, return_runtime=false, CFL=0.5f0);
cont_baseline_swegpu_1st = SWEGPU.SWE_solver(cells, edges, T2, initial_cont; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, return_runtime=false, CFL=0.5f0);
#using JLD
#save("tmp/SWEGPU/circular_dam_break_n=$(n_baseline)_t=$(T).jld", "res", baseline_swegpu)

N_list = [32, 64, 128, 256, 512]

n_list_test, diffs_test, f_test, baseline_n, ord_conv_test = SWEGPU.convergence_test(cont_baseline_swegpu_1st, n_baseline, N_list, T, u0_cont, "Convergence test first order solver"; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, CFL=0.5f0)

n_list, diffs, f, baseline_n, ord_conv = SWEGPU.convergence_test(circ_baseline_swegpu, n_baseline, N_list, T, u0_circ, "Convergence test first order solver"; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, CFL=0.5f0)

n_list2, diffs2, f2, baseline_n, ord_conv2 = SWEGPU.convergence_test(circ_baseline_swegpu, n_baseline, N_list, T, u0_circ, "Convergence test second order solver"; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, CFL=0.45f0, altmesh=false)

n_list3, diffs3, g, baseline_n, ord_conv3 = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T2, u0_cont, "Convergence test first order solver circular"; 
                                                                    backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, CFL=0.5f0, limiter=0)

n_list4, diffs4, g2, baseline_n, ord_conv4 = SWEGPU.convergence_test(cont_baseline_swegpu, n_baseline, N_list, T2, u0_cont, "Convergence test second order solver circular"; 
                                                                    backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, CFL=0.5f0, limiter=1)

# n_list2, diffs2, f2, baseline_n, ord_conv2 = SWEGPU.convergence_test(N_list, T, u0, "Convergence test second order solver", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=1)

# n_list, diffs, f, baseline_n, ord_conv = SWEGPU.convergence_test(N_list, T, u0, "Convergence test second order solver", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=0)

SWEGPU.L1(cont_baseline_swegpu, cont_baseline_swegpu_1st)
SWEGPU.L1(circ_baseline_swegpu, circ_baseline_swegpu_1st)

save("convergence_test_circ_1st_order.png", f)
save("convergence_test_circ_2nd_order.png", f2)
save("convergence_test_cont_1st_order.png", g)
save("convergence_test_cont_2nd_order.png", g2)
save("circular_dam_break_radial_plot_comparison_2nd_order.png", f2)
