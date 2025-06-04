using JLD
include("../structs.jl")

function save_structures(cells::Vector{Cell{T, T1}}, edges::Vector{Edge{T, T1}}, filename) where {T<:Real, T1<:Integer}
    edge_cell_matrix = make_edge_cell_matrix(edges)
    cell_edge_matrix = make_cell_edge_matrix(cells)
    edge_coordinates = make_edge_coordinates_array(edges)
    cell_pts_array = make_cell_pts_array(cells)

    save("tmp/$filename.jld", "edge_cell_matrix", edge_cell_matrix, "cell_edge_matrix", cell_edge_matrix, 
            "edge_coordinates", edge_coordinates, "cell_pts_array", cell_pts_array)
end

function save_structures(cells, edges, filename) 
    edge_cell_matrix = make_edge_cell_matrix(edges)
    cell_edge_matrix = make_cell_edge_matrix(cells)
    edge_coordinates = make_edge_coordinates_array(edges)
    cell_pts_array = make_cell_pts_array(cells)

    save("tmp/$filename.jld", "edge_cell_matrix", edge_cell_matrix, "cell_edge_matrix", cell_edge_matrix, 
            "edge_coordinates", edge_coordinates, "cell_pts_array", cell_pts_array)
end

function load_structures(filename)
    data = load("tmp/$filename.jld")
    edge_cell_matrix = data["edge_cell_matrix"]
    cell_edge_matrix = data["cell_edge_matrix"]
    edge_coordinates = data["edge_coordinates"]
    cell_pts_array = data["cell_pts_array"]
    
    spaceType = eltype(edge_coordinates)
    indType = eltype(edge_cell_matrix)
    n_edges = size(edge_cell_matrix)[1]
    n_cells = size(cell_edge_matrix)[1]
    println("Loading $n_edges edges and $n_cells cells")
    edges = Vector{Edge{spaceType, indType}}(undef, n_edges)
    cells = Vector{Cell{spaceType, indType}}(undef, n_cells)

    for i in 1:n_edges
        pt1 = Vector{spaceType}([edge_coordinates[i, 1, 1], edge_coordinates[i, 1, 2], edge_coordinates[i, 1, 3]])
        pt2 = Vector{spaceType}([edge_coordinates[i, 2, 1], edge_coordinates[i, 2, 2], edge_coordinates[i, 2, 3]])
        c = Vector{indType}([edge_cell_matrix[i, 1], edge_cell_matrix[i, 2]])
        if c[2] == 0
            edges[i] = Edge(pt1, pt2, c[1])
        else
            edges[i] = Edge(pt1, pt2, c[1], c[2])
        end
    end

    for i in 1:n_cells
        e = Vector{indType}([cell_edge_matrix[i, 1], cell_edge_matrix[i, 2], cell_edge_matrix[i, 3]])
        pts = Matrix{spaceType}(cell_pts_array[i, :, :])
        cells[i] = Cell(e, pts)
    end
    return edges, cells
end

function load_structures_with_topography(filename, topography)
    data = load("tmp/$filename.jld")
    edge_cell_matrix = data["edge_cell_matrix"]
    cell_edge_matrix = data["cell_edge_matrix"]
    edge_coordinates = data["edge_coordinates"]
    cell_pts_array = data["cell_pts_array"]
    
    spaceType = eltype(edge_coordinates)
    indType = eltype(edge_cell_matrix)
    n_edges = size(edge_cell_matrix)[1]
    n_cells = size(cell_edge_matrix)[1]
    println("Loading $n_edges edges and $n_cells cells")
    edges = Vector{Edge{spaceType, indType}}(undef, n_edges)
    cells = Vector{Cell{spaceType, indType}}(undef, n_cells)

    for i in 1:n_edges
        
        pt1 = Vector{spaceType}([edge_coordinates[i, 1, 1], edge_coordinates[i, 1, 2], edge_coordinates[i, 1, 3]])
        pt2 = Vector{spaceType}([edge_coordinates[i, 2, 1], edge_coordinates[i, 2, 2], edge_coordinates[i, 2, 3]])
        pt1[3] = topography([pt1[1], pt1[2]])
        pt2[3] = topography([pt2[1], pt2[2]])
        c = Vector{indType}([edge_cell_matrix[i, 1], edge_cell_matrix[i, 2]])
        if c[2] == 0
            edges[i] = Edge(pt1, pt2, c[1])
        else
            edges[i] = Edge(pt1, pt2, c[1], c[2])
        end
        
    end

    for i in 1:n_cells
        e = Vector{indType}([cell_edge_matrix[i, 1], cell_edge_matrix[i, 2], cell_edge_matrix[i, 3]])
        pts = Matrix{spaceType}(cell_pts_array[i, :, :])
        pts[3, 1] = topography([pts[1, 1], pts[2, 1]])
        pts[3, 2] = topography([pts[1, 2], pts[2, 2]])
        pts[3, 3] = topography([pts[1, 3], pts[2, 3]])
        cells[i] = Cell(e, pts)
    end
    return edges, cells
end