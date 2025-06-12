include("../src/SWEGPU.jl")
using .SWEGPU
using Meshes
using GLMakie
using CUDA

n = 128

# Dette tar lang tid for større n (>512), så jeg lagrer hver gang jeg kjører det og sjekker om den er lagret før jeg genererer
edges, cells = SWEGPU.make_structured_mesh(n, n, Float32, Int64);

u0 = x -> [-0.25f0*(sign((x[1]-0.5f0)^2+ (x[2]-0.5f0)^2 -0.01f0) - 1.f0)+ 1.f0, 0.f0, 0.f0]

quad = SWEGPU.quadrature(u0, cells)

T = 0.022f0

res = SWEGPU.SWE_solver(cells, edges, T, quad; backend="CUDA", bc=SWEGPU.neumannBC, reconstruction=1, return_runtime=false)


# Her er litt visualisering som du kan bruke dersom du tror det er nyttig

#viz_recon = SWEGPU.visualize_reconstruction(res, edges, cells; limiter=1)
#viz(viz_recon)

#SWEGPU.SWEGPU.radial_plots([res], cells, ["SWEGPU"], "test")