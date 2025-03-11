include("read_obj.jl")

function generate_mesh(pts, faces)

    edges = []
    #cells = []

    n_faces = size(faces, 2)
    # Assuming the faces are defined in a counter-clockwise order

    for i in 1:n_faces
        face = faces[:,i]
        for j in 1:3
            pt1 = pts[:, face[j]]
            pt2 = pts[:, face[j%3 + 1]]
            # Check if the edge already exists
            edge_exists = false
            for edge in edges
                if edge.pt2 == pt1 && edge.pt1 == pt2
                    edge_exists = true
                    push!(edge.cells, i)
                    break
                end
            end
            if !edge_exists
                push!(edges, Edge{Float64}(pt1, pt2, i))
            end

        end
    end

    return edges #, cells
end
