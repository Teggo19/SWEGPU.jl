include("../src/SWEGPU.jl")
using .SWEGPU

obj_path = "/home/trygve/Master/Kode/Grids/terrain.obj"
test_path = "/home/trygve/Master/Kode/Grids/simple.obj"

vs, fs = SWEGPU.read_obj(obj_path)

edges = SWEGPU.generate_mesh(vs, fs)

SWEGPU.visualize_edges(edges)

using GLMakie

f = Figure(size=(800, 800))
# make a point
p = Point3f([0, 0, 0])
p1 = Point3f(1, 0, 1)
p2 = Point3f(0, 1, 1)
ps = [p, p1, p2]
lines!(f[2, 1], ps)
lines!(f[2, 1], [p, p2], color = :red)
f