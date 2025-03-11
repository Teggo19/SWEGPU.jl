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
        length = sqrt(sum(diff.^2))
        normal = [diff[2], -diff[1]]
        cells = [cell]
        new(pt1, pt2, length, normal, cells)
    end
end

struct Cell{spaceType}
    edges::Vector{Int32}
    area::spaceType

end
