include("structs.jl")
using Meshes
using GLMakie

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
        #push!(vis_points, Point(cell.centroid[1], cell.centroid[2], cell.centroid[3]))
        p1 = (cell.points[1, 1], cell.points[2, 1], cell.points[3, 1])
        p2 = (cell.points[1, 2], cell.points[2, 2], cell.points[3, 2])
        p3 = (cell.points[1, 3], cell.points[2, 3], cell.points[3, 3])
        push!(vis_cells, Triangle(p1, p2, p3))

    end
    
    return vis_cells
end



function visualize_cells_2D(cells)
    n_cells = length(cells)

    vis_points = []
    vis_cells = []

    for i in 1:n_cells
        cell = cells[i]
        #push!(vis_points, Point(cell.centroid[1], cell.centroid[2], cell.centroid[3]))
        p1 = (cell.points[1, 1], cell.points[2, 1])
        p2 = (cell.points[1, 2], cell.points[2, 2])
        p3 = (cell.points[1, 3], cell.points[2, 3])
        push!(vis_cells, Triangle(p1, p2, p3))

    end
    
    return vis_cells
end

function visualize_water(U, cells)
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

function radial_plot(results, cells)
    center = (0.5, 0.5)

    cell_centres = [cell.centroid for cell in cells]

    distances = [sqrt((cell_centre[1]-center[1])^2 + (cell_centre[2]-center[2])^2) for cell_centre in cell_centres]

    points = Point2f.(distances, results[:, 1])
    scatter(points)
end

function radial_plots(results, cells, names, title)
    f = Figure()
    ax1 = Axis(f[1, 1], title=title, xlabel = L"r", ylabel="h")
    ax2 = Axis(f[2, 1], xlabel=L"r", ylabel=L"\sqrt{hu^2 + hv^2}")
    #ax3 = Axis(f[3, 1], title="hv")
    
    center = (0.5, 0.5)

    cell_centres = [cell.centroid for cell in cells]

    distances = [sqrt((cell_centre[1]-center[1])^2 + (cell_centre[2]-center[2])^2) for cell_centre in cell_centres]

    colors = [:red, :blue, :green, :purple, :orange]
    
    for i in 1:length(results)
        points = Point2f.(distances, results[i][:, 1])
        scatter!(ax1, points, color=colors[i], label=names[i], markersize = 2)

        points = Point2f.(distances, sqrt.(results[i][:, 2].^2 .+ results[i][:, 3].^2))
        scatter!(ax2, points, color=colors[i], label=names[i], markersize = 2)

        #points = Point2f.(distances, results[i][:, 3])
        #scatter!(ax3, points, color=colors[i], label=names[i])

    end
    f[1, 2] = Legend(f, ax2, "Plots")
    return f
end

function visualize_reconstruction(U, edges, cells; recon_type=1, limiter=0)
    n_cells = length(cells)
    recon_gradients = make_reconstructions(cells, edges, U; limiter=limiter)
    vis_recon = []

    for i in 1:n_cells
        cell = cells[i]
        h = U[i, recon_type] + cell.centroid[3]
        c = cell.centroid
        h1 = h + recon_gradients[i, recon_type, 1]*(cell.points[1, 1]-c[1]) + recon_gradients[i, recon_type, 2]*(cell.points[2, 1]-c[2])
        h2 = h + recon_gradients[i, recon_type, 1]*(cell.points[1, 2]-c[1]) + recon_gradients[i, recon_type, 2]*(cell.points[2, 2]-c[2])
        h3 = h + recon_gradients[i, recon_type, 1]*(cell.points[1, 3]-c[1]) + recon_gradients[i, recon_type, 2]*(cell.points[2, 3]-c[2])

        p1 = (cell.points[1, 1], cell.points[2, 1], h1)
        p2 = (cell.points[1, 2], cell.points[2, 2], h2)
        p3 = (cell.points[1, 3], cell.points[2, 3], h3)

        push!(vis_recon, Triangle(p1, p2, p3))
    end

    return vis_recon
end


include("Solver/reconstruction.jl")
function make_reconstructions(cells, edges, initial; limiter=0)
    spaceType = eltype(cells[1].centroid)

    n_cells = length(cells)
    

    U = deepcopy(initial)
    
    # Holds information of what cells each edge is connected to
    edge_cell_matrix = make_edge_cell_matrix(edges)
    centroids = hcat([cell.centroid for cell in cells]...)'


    edge_coordinates = make_edge_center_matrix(edges)

    # Holds information of what edges each cell is connected to
    cell_edge_matrix = make_cell_edge_matrix(cells)
    

    recon_gradient = zeros(spaceType, n_cells, 3, 2)
    
    dev = CPU()

    update_reconstruction!(dev, 64)(U, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates, limiter, ndrange=n_cells)
    return recon_gradient
end

function visualize_heatmap(U, cells; min_max = [0, 2])
    n_cells = length(cells)

    vis_list = []
    colors = Vector{Float64}()

    for i in 1:n_cells
        cell = cells[i]
        p1 = (cell.points[1, 1], cell.points[2, 1])
        p2 = (cell.points[1, 2], cell.points[2, 2])
        p3 = (cell.points[1, 3], cell.points[2, 3])
        # find the color according to the value of U[i, 1]
        
        # create a triangle with the color
        triangle = Triangle(p1, p2, p3)
        push!(vis_list, triangle)
        push!(colors, U[i]/min_max[2])

    end
    return vis_list, colors
end


function visualize_height(U, cells, edges)
    n_cells = length(cells)
    
    vis_heights = []

    for i in 1:n_cells
        cell = cells[i]
        h = U[i, 1] + cell.centroid[3]
        p1 = (cell.points[1, 1], cell.points[2, 1], h)
        p2 = (cell.points[1, 2], cell.points[2, 2], h)
        p3 = (cell.points[1, 3], cell.points[2, 3], h)
        push!(vis_heights, Triangle(p1, p2, p3))
    end
    #=
    n_edges = length(edges)
    for i in 1:n_edges
        edge = edges[i]
        if length(edge.cells) == 2
            h2 = U[edge.cells[2]] + cells[edge.cells[2]].centroid[3]
        
            h1 = U[edge.cells[1]] + cells[edge.cells[1]].centroid[3]
            
            if h1 != h2
                p1 = (edge.pt1[1], edge.pt1[2], h1)
                p2 = (edge.pt2[1], edge.pt2[2], h1)
                p3 = (edge.pt2[1], edge.pt2[2], h2)
                p4 = (edge.pt1[1], edge.pt1[2], h2)
                push!(vis_heights, Quadrangle(p1, p2, p3, p4))
            end
        end
    end=#
    

    return vis_heights
end