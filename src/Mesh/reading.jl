include("../structs.jl")
using ProgressMeter

function read_obj(path; spaceType=Float32, indType=Int64)
    vertices = Vector{Vector{spaceType}}()
    faces = Vector{Vector{indType}}()

    for line in eachline(path)
        if startswith(line, "v ")
            push!(vertices, map(x -> parse(spaceType, x), split(line[3:end])))
        elseif startswith(line, "f ")
            push!(faces, map(x -> parse(indType, x), split(line[3:end])))
        end
    end
    vertices = hcat(vertices...)
    faces = hcat(faces...)
    #transpose(vertices)
    #transpose(faces)
    return vertices, faces
end

function generate_mesh(pts, faces, spaceType, indType)
    #=
    Generate the edges and cells from points and faces.
    =#

    edges = Vector{Edge{spaceType}}()
    cells = Vector{Cell{spaceType}}()

    non_completed_edges=Vector{indType}()

    n_faces = size(faces, 2)
    n_pts = size(pts, 2)

    pts_overview = Vector{Vector{indType}}(undef, n_pts)
    for i in 1:n_pts
        pts_overview[i] = []
    end

    # Assuming the faces are defined in a counter-clockwise order
    p = Progress(n_faces; dt=0.2)
    for i in 1:n_faces
        cell_edges = Vector{indType}()
        face = faces[:,i]
        for j in 1:3
            pt1_index = face[j]
            pt2_index = face[j%3 + 1]
            pt1 = pts[:, pt1_index]
            pt2 = pts[:, pt2_index]
            # Check if the edge already exists

            pts_array = [pts_overview[pt1_index]; pts_overview[pt2_index]]
            edge_exists = false
            for k in pts_array
                edge = edges[k]
                if edge.pt2 == pt1 && edge.pt1 == pt2
                    edge_exists = true
                    push!(edge.cells, i)
                    push!(cell_edges, k)
                    break
                end
            end
            if !edge_exists
                push!(edges, Edge(pt1, pt2, i))
                push!(cell_edges, length(edges))
                push!(pts_overview[pt1_index], length(edges))
                push!(pts_overview[pt2_index], length(edges))
            end
            #=
            edge_exists = false
            for (ind, k) in enumerate(non_completed_edges)
                edge = edges[k]
                if edge.pt2 == pt1 && edge.pt1 == pt2
                    edge_exists = true
                    push!(edge.cells, i)
                    push!(cell_edges, k)
                    deleteat!(non_completed_edges, ind)
                    break
                end
            end
            if !edge_exists
                push!(edges, Edge(pt1, pt2, i))
                push!(cell_edges, length(edges))
                push!(non_completed_edges, length(edges))
            end=#

        end
        push!(cells, Cell(cell_edges, hcat(pts[:, face[1]], pts[:, face[2]], pts[:, face[3]])))
        update!(p, i)
    end

    return edges , cells
end
