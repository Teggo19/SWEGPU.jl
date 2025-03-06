import ArchGDAL
import GLMakie, Images
import GeometryBasics
import Meshes

include("src/SWEGPU.jl")
using .SWEGPU

kuba_path = "/home/trygve/Master/Kode/Grids/swimdata/data/kuba"

geoarray_dtm = ArchGDAL.readraster(joinpath(kuba_path, "dtm1", "data", "dtm1.tif"))

grid_dtm = permutedims(geoarray_dtm[:,:,1]) .|> Float64


function read_obj(path)
    vertices = []
    faces = []

    for line in eachline(path)
        if startswith(line, "v ")
            push!(vertices, map(x -> parse(Float64, x), split(line[3:end])))
        elseif startswith(line, "f ")
            push!(faces, map(x -> parse(Int, x), split(line[3:end])))
        end
    end
    vertices = hcat(vertices...)
    faces = hcat(faces...)
    #transpose(vertices)
    #transpose(faces)
    return vertices, faces
end

obj_path = "/home/trygve/Master/Kode/Grids/terrain.obj"
vs, fs = read_obj(obj_path)
size(vs, 2)

colors = [:red, :green, :blue, :yellow, :purple, :orange, :cyan, :magenta, :white, :black]

cs = rand(colors, size(vs, 2))



vertices, faces = make_geometrybasics_vert(vs, fs)
vertices



m = GeometryBasics.mesh(vertices, faces, color = cs)

GLMakie.mesh(m, color = cs)

using Meshes
grid = Meshes.CartesianGrid(10, 10, 10)
viz(grid, showsegments=true, segmentcolor=:teal)
triangles = rand(Triangle, 10, crs=Meshes.Cartesian3D)

viz(triangles, color = 1:10)

vs

function make_triangles(vs, fs)
    triangles = Vector{Triangle}()

    n_faces = size(fs, 2)
    for i in 1:n_faces
        p1 = (vs[1, fs[1, i]], vs[2, fs[1, i]], vs[3, fs[1, i]])
        p2 = (vs[1, fs[2, i]], vs[2, fs[2, i]], vs[3, fs[2, i]])
        p3 = (vs[1, fs[3, i]], vs[2, fs[3, i]], vs[3, fs[3, i]])
        push!(triangles, Triangle(p1, p2, p3))
    end
    return triangles
end
n_faces = size(fs, 2)
triangles = [Triangle((vs[j, fs[1, i]] for j in 1:3), (vs[j, fs[2, i]] for j in 1:3), (vs[j, fs[3, i]] for j in 1:3) ) for i in 1:n_faces]

triangles = make_triangles(vs, fs)

t = Triangle((0, 0, 1), (1, 0, 0), (0, 1, 0))

viz(triangles, showsegments=true, segmentcolor=:red, color =:grey, alpha = 0)