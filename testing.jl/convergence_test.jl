include("../src/SWEGPU.jl")
using .SWEGPU
include("../comparing_with_SinFVM/file_helper.jl")
include("../comparing_with_SinFVM/converting.jl")

T = 0.022f0
n_baseline = 2048
baseline_quad = read_from_file("circular_dam_break_n=$(n_baseline)_t=$(T)")
baseline_sinfvm = cartesian_to_triangular(baseline_quad)
# circular dam break
#u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]
u0 = x -> [0.5f0 + 0.2f0*exp((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2), 0.f0, 0.f0]

edges, cells = SWEGPU.make_structured_mesh(n_baseline, n_baseline, Float32, Int64);
initial = SWEGPU.quadrature(u0, cells);

baseline_swegpu = SWEGPU.SWE_solver(cells, edges, T, initial; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false);
#using JLD
#save("tmp/SWEGPU/circular_dam_break_n=$(n_baseline)_t=$(T).jld", "res", baseline_swegpu)

N_list = [32, 64, 128, 256, 512]

n_list, diffs, f, baseline_n, ord_conv = SWEGPU.convergence_test(baseline_swegpu, n_baseline, N_list, T, u0, "Convergence test first order solver"; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=0)

n_list2, diffs2, f2, baseline_n, ord_conv2 = SWEGPU.convergence_test(baseline_swegpu, n_baseline, N_list, T, u0, "Convergence test first order solver"; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1)

n_list2, diffs2, f2, baseline_n, ord_conv2 = SWEGPU.convergence_test(N_list, T, u0, "Convergence test second order solver", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=1)

n_list, diffs, f, baseline_n, ord_conv = SWEGPU.convergence_test(N_list, T, u0, "Convergence test second order solver", backend="CUDA", bc=SWEGPU.wallBC, reconstruction=0)