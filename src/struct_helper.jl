include("structs.jl")




function make_edge_cell_matrix(edges)
    
    res = zeros(eltype(edges[1].cells), length(edges), 2)
    for (i, edge) in enumerate(edges)
        res[i, 1] = edge.cells[1]
        if length(edge.cells) == 2
            res[i, 2] = edge.cells[2]
        end
    end
    return res
end

function make_cell_edge_matrix(cells)
    res = zeros(eltype(cells[1].edges), length(cells), 3)
    for (i, cell) in enumerate(cells)
        if length(cell.edges) == 3
            res[i, :] = cell.edges
        else
            # throw error
            error("Cell at index $i does not have 3 edges")

            #res[i, 1:length(cell.edges)] = cell.edges
        end
    end
    return res
end

function make_normal_matrix(edges) 
    res = zeros(eltype(edges[1].normal), length(edges), 2)
    for (i, edge) in enumerate(edges)
        res[i, :] = edge.normal
    end
    return res
end


function cell_centres(cells) 
    res = zeros(eltype(cells[1].centroid), length(cells), 3)
    for (i, cell) in enumerate(cells)
        res[i, :] = cell.centroid
    end
    return res
end

function compute_diameter(pts::Matrix{T}, area::T) where {T<:Real}
    pt1 = pts[1:2, 1]
    pt2 = pts[1:2, 2]
    pt3 = pts[1:2, 3]

    l1 = sqrt(sum((pt1-pt2).^2))
    l2 = sqrt(sum((pt1-pt3).^2))
    l3 = sqrt(sum((pt2-pt3).^2))

    s=0.5*(l1+l2+l3)

    return 2*area/s
end

function compute_cell_gradient(pts::Matrix{T}) where {T<:Real}
    pt1 = pts[1:3, 1]
    pt2 = pts[1:3, 2]
    pt3 = pts[1:3, 3]
    # Compute the gradient of the cell
    grad = zeros(T, 2)
    grad[1] = ((pt2[3] - pt1[3])*(pt3[2] - pt1[2]) - (pt3[3] - pt1[3])*(pt2[2] - pt1[2]))/((pt2[1] - pt1[1])*(pt3[2] - pt1[2]) - (pt3[1] - pt1[1])*(pt2[2] - pt1[2]))
    grad[2] = ((pt2[3] - pt1[3])*(pt3[1] - pt1[1]) - (pt3[3] - pt1[3])*(pt2[1] - pt1[1]))/((pt2[2] - pt1[2])*(pt3[1] - pt1[1]) - (pt3[2] - pt1[2])*(pt2[1] - pt1[1]))
    return grad
end

function make_edge_center_matrix(edges)
    res = zeros(eltype(edges[1].pt1), length(edges), 3)
    for (i, edge) in enumerate(edges)
        res[i, 1] = 0.5*(edge.pt1[1] + edge.pt2[1])
        res[i, 2] = 0.5*(edge.pt1[2] + edge.pt2[2])
        res[i, 3] = 0.5*(edge.pt1[3] + edge.pt2[3])
    end
    return res
end

function make_edge_coordinates_array(edges)
    res = zeros(eltype(edges[1].pt1), length(edges), 2, 3)
    for (i, edge) in enumerate(edges)
        res[i, 1, 1] = edge.pt1[1]
        res[i, 1, 2] = edge.pt1[2]
        res[i, 1, 3] = edge.pt1[3]
        res[i, 2, 1] = edge.pt2[1]
        res[i, 2, 2] = edge.pt2[2]
        res[i, 2, 3] = edge.pt2[3]
    end
    return res
end

function make_cell_pts_array(cells) 
    res = zeros(eltype(cells[1].points), length(cells), 3, 3)
    for (i, cell) in enumerate(cells)
        res[i, :, :] = cell.points
    end
    return res
end