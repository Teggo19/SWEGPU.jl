include("../read_obj.jl")
include("../structs.jl")
using ProgressMeter

function generate_mesh(pts, faces)
    #=
    Generate the edges and cells from points and faces.
    =#

    edges = Vector{Edge{Float64}}()
    cells = Vector{Cell{Float64}}()

    non_completed_edges=Vector{Int64}()

    n_faces = size(faces, 2)
    # Assuming the faces are defined in a counter-clockwise order
    p = Progress(n_faces; dt=0.2)
    for i in 1:n_faces
        cell_edges = Vector{Int64}()
        face = faces[:,i]
        for j in 1:3
            pt1 = pts[:, face[j]]
            pt2 = pts[:, face[j%3 + 1]]
            # Check if the edge already exists
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
                push!(edges, Edge{Float64}(pt1, pt2, i))
                push!(cell_edges, length(edges))
                push!(non_completed_edges, length(edges))
            end

        end
        push!(cells, Cell{Float64}(cell_edges, hcat(pts[:, face[1]], pts[:, face[2]], pts[:, face[3]])))
        update!(p, i)
    end

    return edges , cells
end
