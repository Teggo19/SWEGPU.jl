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

    return vertices, faces
end