include("../src/SWEGPU.jl")
using .SWEGPU
using GLMakie
using Meshes
using Polynomials


T = 0.0f0
u0_cont = x -> [0.5f0*exp(-((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2)/0.02)+0.02, 0.f0, 0.f0]

n_baseline = 2048

edges_baseline, cells_baseline = SWEGPU.make_structured_mesh(n_baseline, n_baseline, Float64, Int64);
initial_baseline = SWEGPU.quadrature(u0_cont, cells_baseline; order=4);

N_list = [32, 64, 128, 256, 512]

diffs2 = zeros(length(N_list))
for (i, n) in enumerate(N_list)
    edges, cells = SWEGPU.make_structured_mesh(n, n, Float64, Int64);
    initial = SWEGPU.quadrature(u0_cont, cells; order=4);
    recon = SWEGPU.make_reconstructions(cells, edges, initial; limiter=1);
    res, _ = SWEGPU.refine_structured_grid(initial, n, n_baseline, recon)

    diffs2[i] = SWEGPU.L1(res, initial_baseline)
end

ord_conv = log.(2, diffs[1:end-1] ./ diffs[2:end])

ord_conv2 = log.(2, diffs2[1:end-1] ./ diffs2[2:end])
for (i, N) in enumerate(N_list)
    if i == 1
        c = 0
        c2 = 0
    else
        c = ord_conv[i-1]
        c2 = ord_conv2[i-1]
    end
    println("\$2^{$(i+4)}\$ & $(round(diffs[i], digits=5))& $(round(c, digits=5)) & $(round(diffs2[i], digits=5)) & $(round(c2, digits=5))\\\\")
end