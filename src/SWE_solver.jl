using KernelAbstractions
using ProgressMeter

include("structs.jl")
include("FVM.jl")

function SWE_solver(cells, edges, T, initial)
    # TODO: Dynamic type allocation to allow for Automatic Differentiation
    dev = CPU()
    # Set initial conditions
    n_edges = length(edges)
    n_cells = length(cells)
    
    t = 0

    U = deepcopy(initial)
    # h = initial[:, 0]
    # hu = initial[:, 1]
    # hv = initial[:, 2]
    
    # Holds information of what cells each edge is connected to
    edge_cell_matrix = make_edge_cell_matrix(edges)
    centroids = [cell.centroid for cell in cells]
    areas = [cell.area for cell in cells]
    diameters = [cell.diameter for cell in cells]
    edge_lengths = [edge.length for edge in edges]

    # Holds information of what edges each cell is connected to
    cell_edge_matrix = make_cell_edge_matrix(cells)

    fluxes = zeros(Float64, n_edges, 3)
    normal_matrix = make_normal_matrix(edges)

    max_dt_array = ones(Float64, length(cells))
    # Loop over time
    p = Progress(Int64(ceil(T*3840)); dt=0.1)
    
    CFL = 0.1
    while t < T
        # Loop over edges
        update_fluxes!(dev, 32)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, ndrange=n_edges)
        
        #println("fluxes: \n", fluxes)
        dt = T - t

        max_dt = minimum(max_dt_array)

        dt = min(dt, CFL*max_dt)
        #println("dt : ", dt)
        t += dt
        # Loop over cells
        update_values!(dev, 32)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, ndrange=n_cells)

        update!(p, Int64(ceil(t*3840)))
    end
    finish!(p)

    return U
end



