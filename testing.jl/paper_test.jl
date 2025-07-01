include("../src/SWEGPU.jl")
using .SWEGPU
using GLMakie
using Meshes

T = 0.07f0



top_func = x -> 0.5*exp(-50.f0*(x[1]-0.5f0)^2 - 50.f0*(x[2]-0.5f0)^2)
u0 = x -> [1.f0, 0.3f0, 0.f0]
n_baseline = 2048

edges_baseline, cells_baseline = SWEGPU.make_structured_mesh_with_topography(n_baseline, n_baseline, Float32, Int64, top_func);
initial = SWEGPU.quadrature(u0, cells_baseline);

baseline = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, return_runtime=false, CFL=0.5f0);


N_list = [32, 64, 128, 256, 512]

n_list, diffs, f, baseline_n, ord_conv = SWEGPU.convergence_test(baseline, n_baseline, N_list, T, u0, "Convergence test second order solver"; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, limiter=0, CFL=0.5f0, top_func=top_func)

n_list, diffs2, f2, baseline_n, ord_conv2 = SWEGPU.convergence_test(baseline, n_baseline, N_list, T, u0, "Convergence test first order solver"; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=0, CFL=0.5f0, top_func=top_func, limiter=2)


n_list, diffs3, f3, baseline_n, ord_conv3 = SWEGPU.convergence_test(baseline, n_baseline, N_list, T, u0, "Convergence test first order solver"; backend="CUDA", bc=SWEGPU.neumannBC, time_stepper=1, CFL=0.5f0, top_func=top_func, limiter=1)


#save("tmp/convergence_test/convergence_test_topography_2nd_order.png", f)
#save("tmp/convergence_test/convergence_test_topography_1st_order.png", f2)

#u0_circ = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

#edges_baseline, cells_baseline = SWEGPU.make_structured_mesh(n_baseline, n_baseline, Float32, Int64);
#initial_circ = SWEGPU.quadrature(u0_circ, cells_baseline);

#baseline_circ = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_circ; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false, CFL=0.5f0);

#n_list, diffs_circ, f_circ, baseline_n, ord_conv_circ = SWEGPU.convergence_test(baseline_circ, n_baseline, N_list, T, u0_circ, "Convergence test second order solver"; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, limiter=0, CFL=0.5f0)
#n_list_circ, diffs2_circ, f2_circ, baseline_n_circ, ord_conv2_circ = SWEGPU.convergence_test(baseline_circ, n_baseline, N_list, T, u0_circ, "Convergence test first order solver"; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0, CFL=0.5f0)

#save("tmp/convergence_test/convergence_test_circular_dam_break_2nd_order.png", f_circ)
#save("tmp/convergence_test/convergence_test_circular_dam_break_1st_order.png", f2_circ)


for (i, N) in enumerate(N_list)
    a = diffs3[i]
    c = diffs2[i]
    e = diffs[i]
    if i == 1
        b = 0
        d = 0
        f = 0
    else
        b = log(2, diffs3[i-1]/diffs3[i])
        d = log(2, diffs2[i-1]/diffs2[i])
        f = log(2, diffs[i-1]/diffs[i])
    end 
    println("\$2^{$(i+4)}\$ & $(round(a, digits=6))& $(round(b, digits=4)) & $(round(c, digits=6)) & $(round(d, digits=4)) & $(round(e, digits=6)) & $(round(f, digits=4))  \\\\")
end