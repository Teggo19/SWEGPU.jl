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
    # Reshape to 3xN matrix
    vertices = hcat(vertices...)
    faces = hcat(faces...)

    return vertices, faces
end

using Meshes
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