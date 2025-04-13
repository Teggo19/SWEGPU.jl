

struct Edge{spaceType<:Real, indType<:Integer} 
    pt1::Vector{spaceType}
    pt2::Vector{spaceType}
    length::spaceType
    normal::Vector{spaceType}
    # Maybe have this int type be variable as well
    cells::Vector{indType}  # opposite side of normal
    # need to figure out how to find cell1 and cell2
    function Edge(pt1::Vector{spaceType}, pt2::Vector{spaceType}, cell::indType) where {spaceType<:Real, indType<:Integer}
        diff = pt2 - pt1
        length = sqrt(diff[1]^2+diff[2]^2)
        normal = [diff[2], -diff[1]] ./ (length)
        cells = [cell]
        new{spaceType, indType}(pt1, pt2, length, normal, cells)
    end
end

struct Cell{spaceType<:Real, indType<:Integer}
    edges::Vector{indType}
    centroid::Vector{spaceType}
    points::Matrix{spaceType}
    area::spaceType
    diameter::spaceType

    function Cell(edges::Vector{indType}, pts::Matrix{spaceType}) where {spaceType<:Real, indType<:Integer}
        centroid = (pts[1:3, 1]+ pts[1:3, 2]+ pts[1:3, 3])/3
        area = convert(spaceType, 0.5*abs((pts[1, 1]-pts[1, 3])*(pts[2, 2]-pts[2, 1])-(pts[1, 1]-pts[1, 2])*(pts[2, 3]-pts[2, 1])))
        d = compute_diameter(pts, area)
        new{spaceType, indType}(copy(edges), centroid, pts, area, d)
    end
end


function make_edge_cell_matrix(edges::Vector{Edge{spaceType}}) where {spaceType<:Real}
    
    res = zeros(eltype(edges[1].cells), length(edges), 2)
    for (i, edge) in enumerate(edges)
        res[i, 1] = edge.cells[1]
        if length(edge.cells) == 2
            res[i, 2] = edge.cells[2]
        end
    end
    return res
end

function make_cell_edge_matrix(cells::Vector{Cell{spaceType}}) where {spaceType<:Real}
    res = zeros(eltype(cells[1].edges), length(cells), 3)
    for (i, cell) in enumerate(cells)
        if length(cell.edges) == 3
            res[i, :] = cell.edges
        else
            res[i, 1:length(cell.edges)] = cell.edges
        end
    end
    return res
end

function make_normal_matrix(edges::Vector{Edge{T}}) where {T<:Real}
    res = zeros(T, length(edges), 2)
    for (i, edge) in enumerate(edges)
        res[i, :] = edge.normal
    end
    return res
end


function cell_centres(cells::Vector{Cell{T}}) where {T<:Real}
    res = zeros(T, length(cells), 3)
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