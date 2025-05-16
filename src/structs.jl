struct Edge{spaceType<:Real, indType<:Integer} 
    pt1::Vector{spaceType}
    pt2::Vector{spaceType}
    length::spaceType
    normal::Vector{spaceType}
    # Maybe have this int type be variable as well
    cells::Vector{indType}  # The first cell is the one that the normal points out of, 
                            # the second cell is the one that the normal points into (if it exists)
    # need to figure out how to find cell1 and cell2
    function Edge(pt1::Vector{spaceType}, pt2::Vector{spaceType}, cell::indType) where {spaceType<:Real, indType<:Integer}
        diff = pt2 - pt1
        length = sqrt(diff[1]^2+diff[2]^2)
        normal = [diff[2], -diff[1]] ./ (length)
        cells = [cell]
        new{spaceType, indType}(pt1, pt2, length, normal, cells)
    end
    function Edge(pt1::Vector{spaceType}, pt2::Vector{spaceType}, cell1::indType, cell2::indType) where {spaceType<:Real, indType<:Integer}
        diff = pt2 - pt1
        length = sqrt(diff[1]^2+diff[2]^2)
        normal = [diff[2], -diff[1]] ./ (length)
        cells = [cell1, cell2]
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

