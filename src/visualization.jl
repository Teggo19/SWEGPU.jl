include("structs.jl")
using Meshes
#using GLMakie

function visualize_edges(edges)
    #figure = Figure(size=(1000, 1000))
    n_edges = length(edges)
    interior_segments = []
    boundary_segments = []
    #colors = [RGBAf(i/n_edges, 0, 1 - i/n_edges, 1) for i in 1:n_edges]
    for i in 1:n_edges
        edge = edges[i]
        p1 = (edge.pt1[1], edge.pt1[2], edge.pt1[3])
        p2 = (edge.pt2[1], edge.pt2[2], edge.pt2[3])
        #p1 = Point3f(edge.pt1)
        #p2 = Point3f(edge.pt2)
        if length(edge.cells) == 1
            push!(boundary_segments, Segment(p1, p2))
        else
            push!(interior_segments, Segment(p1, p2))
        end
    end
    return interior_segments, boundary_segments
end

function visualize_cells(cells)
    n_cells = length(cells)

    vis_points = []
    vis_cells = []

    for i in 1:n_cells
        cell = cells[i]
        push!(vis_points, Point(cell.centroid[1], cell.centroid[2], cell.centroid[3]))
        p1 = (cell.points[1, 1], cell.points[2, 1], cell.points[3, 1])
        p2 = (cell.points[1, 2], cell.points[2, 2], cell.points[3, 2])
        p3 = (cell.points[1, 3], cell.points[2, 3], cell.points[3, 3])
        push!(vis_cells, Triangle(p1, p2, p3))

    end
    
    return vis_cells, vis_points
end

function visualize_height(U, cells, edges)
    n_cells = length(cells)
    
    vis_heights = []

    for i in 1:n_cells
        cell = cells[i]
        h = U[i, 1]

        p1 = (cell.points[1, 1], cell.points[2, 1], h)
        p2 = (cell.points[1, 2], cell.points[2, 2], h)
        p3 = (cell.points[1, 3], cell.points[2, 3], h)
        push!(vis_heights, Triangle(p1, p2, p3))
    end

    n_edges = length(edges)
    for i in 1:n_edges
        edge = edges[i]
        if length(edge.cells) == 2
            h2 = U[edge.cells[2]]
        
            h1 = U[edge.cells[1]]
            
            if h1 != h2
                p1 = (edge.pt1[1], edge.pt1[2], h1)
                p2 = (edge.pt2[1], edge.pt2[2], h1)
                p3 = (edge.pt2[1], edge.pt2[2], h2)
                p4 = (edge.pt1[1], edge.pt1[2], h2)
                push!(vis_heights, Quadrangle(p1, p2, p3, p4))
            end
        end
    end
    

    return vis_heights
end

function visualize_water(U, cells::Vector{Cell{Float64}})
    n_cells = length(cells)

    vis_water = []

    for i in 1:n_cells
        cell = cells[i]
        h = U[i, 1]
        p1 = (cell.points[1, 1], cell.points[2, 1], h)
        p2 = (cell.points[1, 2], cell.points[2, 2], h)
        p3 = (cell.points[1, 3], cell.points[2, 3], h)

        p4 = (cell.points[1, 1], cell.points[2, 1], cell.points[3, 1])
        p5 = (cell.points[1, 2], cell.points[2, 2], cell.points[3, 2])
        p6 = (cell.points[1, 3], cell.points[2, 3], cell.points[3, 3])

        push!(vis_water, Wedge(p1, p2, p3, p4, p5, p6))
    end

    return vis_water
end