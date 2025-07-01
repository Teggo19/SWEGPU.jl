include("../src/SWEGPU.jl")
using .SWEGPU
using GLMakie
using Meshes


T = 0.22f0
#u0_cont = x -> [0.5f0*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02)+0.02, 0.f0, 0.f0]
u0_circ = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.1f0) - 1.0f0)+ 1.f0, 0.f0, 0.f0]
n_baseline = 4


edges_baseline, cells_baseline = SWEGPU.make_structured_mesh(n_baseline, n_baseline, Float32, Int64);

initial_cont = SWEGPU.quadrature(u0_circ, cells_baseline)

viz_initial = SWEGPU.visualize_height(initial_cont, cells_baseline, edges_baseline)
viz(viz_initial, showsegments=true)
viz!(viz_initial[10], color=:green)

gr1 = [1, 2, 7, 8, 25, 26, 31, 32]
gr2 = [3, 6, 9, 16, 17, 24, 27, 30]
gr3 = [4, 5, 10, 15, 18, 23, 28, 29]
gr4 = [11, 12, 13, 14, 19, 20, 21, 22]
groups = [gr1, gr2, gr3, gr4]
for i in gr1
    viz!(viz_initial[i], color=:blue)
end
for i in gr2
    viz!(viz_initial[i], color=:green)
end
for i in gr3
    viz!(viz_initial[i], color=:orange)
end
for i in gr4
    viz!(viz_initial[i], color=:purple)
end

res_00 = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_cont; backend="cpu", time_stepper=0, limiter=2, CFL=0.4f0)
res_01 = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_cont; backend="cpu", time_stepper=0, limiter=0, CFL=0.4f0)
res_02 = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_cont; backend="cpu", time_stepper=0, limiter=1, CFL=0.4f0)
res_10 = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_cont; backend="cpu", time_stepper=1, limiter=2, CFL=0.4f0)
res_11 = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_cont; backend="cpu", time_stepper=1, limiter=0, CFL=0.4f0);
res_12 = SWEGPU.SWE_solver(cells_baseline, edges_baseline, T, initial_cont; backend="cpu", time_stepper=1, limiter=1, CFL=0.4f0)


SWEGPU.radial_plots([res_00, res_11], cells_baseline, ["res_00", "res_11"]; title="Symmetry test")

for i in 1:div(length(cells_baseline), 2)
    if symm[i, 1] != symm[end-i+1, 1]
        println("Symmetry broken at cell $i")
    end
end

check_symmetry(res_12, groups)

function check_symmetry(res, groups)
    for group in groups
        group_res = res[group, :]
        max_val = maximum(group_res[:, 1])
        min_val = minimum(group_res[:, 1])
        if abs(max_val - min_val) > 1e-8
            println("Symmetry broken in group $(group) with max: $max_val and min: $min_val, maxindex: $(group[argmax(group_res[:, 1])]), minindex: $(group[argmin(group_res[:, 1])])")
        else
            println("Group $(group) is symmetric with value: $max_val")
        end
    end
end

vis_recon = SWEGPU.visualize_reconstruction(initial_cont, edges_baseline, cells_baseline; recon_type=1, limiter=1)
viz(vis_recon, showsegments=true)