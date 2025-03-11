include("structs.jl")
#using Meshes
using GLMakie

function visualize_edges(edges)
    figure = Figure(size=(1000, 1000))
    n_edges = length(edges)
    viz_edges = []
    colors = [RGBAf(i/n_edges, 0, 1 - i/n_edges, 1) for i in 1:n_edges]
    for i in 1:n_edges
        edge = edges[i]
        p1 = Point3f(edge.pt1)
        p2 = Point3f(edge.pt2)
        if i == 1
            lines(figure[1, 1], [p1, p2], color = colors[i])
        else
            lines!(figure[1, 1], [p1, p2], color = colors[i])
        end
        
    end
    display(figure)
    return
end
