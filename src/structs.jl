struct Edge{spaceType} 
    pt1::Vector{spaceType}
    pt2::Vector{spaceType}
    length::spaceType
    normal::Vector{spaceType}
    # Maybe have this int type be variable as well
    cells::Vector{Int64}  # opposite side of normal
    # need to figure out how to find cell1 and cell2
    function Edge{Float64}(pt1::Vector{Float64}, pt2::Vector{Float64}, cell::Int64)
        diff = pt2 - pt1
        length = sqrt(diff[1]^2+diff[2]^2)
        normal = [diff[2], -diff[1]] ./ (length)
        cells = [cell]
        new(pt1, pt2, length, normal, cells)
    end
end

struct Cell{spaceType}
    edges::Vector{Int64}
    centroid::Vector{spaceType}
    points::Matrix{spaceType}
    area::spaceType
    diameter::spaceType

    function Cell{Float64}(edges::Vector{Int64}, pts::Matrix{Float64})
        centroid = (pts[1:3, 1]+ pts[1:3, 2]+ pts[1:3, 3])/3
        area = 0.5*abs((pts[1, 1]-pts[1, 3])*(pts[2, 2]-pts[2, 1])-(pts[1, 1]-pts[1, 2])*(pts[2, 3]-pts[2, 1]))
        d = compute_diameter(pts, area)
        new(edges, centroid, pts, area, d)
    end
end


function make_edge_cell_matrix(edges::Vector{Edge{Float64}})
    res = zeros(Int64, length(edges), 2)
    for (i, edge) in enumerate(edges)
        res[i, 1] = edge.cells[1]
        if length(edge.cells) == 2
            res[i, 2] = edge.cells[2]
        end
    end
    return res
end

function make_cell_edge_matrix(cells::Vector{Cell{Float64}})
    res = zeros(Int64, length(cells), 3)
    for (i, cell) in enumerate(cells)
        if length(cell.edges) == 3
            res[i, :] = cell.edges
        else
            res[i, 1:length(cell.edges)] = cell.edges
        end
    end
    return res
end

function make_normal_matrix(edges::Vector{Edge{Float64}})
    res = zeros(Float64, length(edges), 2)
    for (i, edge) in enumerate(edges)
        res[i, :] = edge.normal
    end
    return res
end


function cell_centres(cells::Vector{Cell{Float64}})
    res = zeros(Float64, length(cells), 3)
    for (i, cell) in enumerate(cells)
        res[i, :] = cell.centroid
    end
    return res
end

function compute_diameter(pts::Matrix{Float64}, area::Float64)
    pt1 = pts[1:2, 1]
    pt2 = pts[1:2, 2]
    pt3 = pts[1:2, 3]

    l1 = sqrt(sum((pt1-pt2).^2))
    l2 = sqrt(sum((pt1-pt3).^2))
    l3 = sqrt(sum((pt2-pt3).^2))

    s=0.5*(l1+l2+l3)

    return 2*area/s
end