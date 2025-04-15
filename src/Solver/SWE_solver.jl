using KernelAbstractions
using ProgressMeter
using CUDA

include("../structs.jl")
include("FVM.jl")

function SWE_solver(cells, edges, T, initial; backend="CPU")
    # TODO: Dynamic type allocation to allow for Automatic Differentiation
    
    # Set initial conditions
    n_edges = length(edges)
    n_cells = length(cells)
    
    t = 0

    U = deepcopy(initial)
    
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

    max_dt_array = ones(Float64, n_cells)

    if backend == "CUDA"
        fluxes = CuArray(fluxes)
        U = CuArray(U)
        edge_cell_matrix = CuArray(edge_cell_matrix)
        cell_edge_matrix = CuArray(cell_edge_matrix)
        normal_matrix = CuArray(normal_matrix)
        edge_lengths = CuArray(edge_lengths)
        max_dt_array = CuArray(max_dt_array)
        diameters = CuArray(diameters)
        areas = CuArray(areas)

        dev = get_backend(U)
    
    else
        dev = CPU()
    end

    # Loop over time
    p = Progress(Int64(ceil(T*3840)); dt=0.1)
    
    CFL = 0.25
    while t < T
        # Loop over edges
        update_fluxes!(dev, 64)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, ndrange=n_edges)
        KernelAbstractions.synchronize(dev)
        dt = T - t
        if backend =="CUDA"
            max_dt = CUDA.minimum(max_dt_array)
            KernelAbstractions.synchronize(dev)
        else
            max_dt = minimum(max_dt_array)
        end
        dt = min(dt, CFL*max_dt)
        #println("dt : ", dt)
        t += dt
        # Loop over cells
        update_values!(dev, 64)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, max_dt_array, ndrange=n_cells)
        KernelAbstractions.synchronize(dev)
        update!(p, Int64(ceil(t*3840)))
    end
    finish!(p)
    if backend == "CUDA"
        U = collect(U)
    end
    return U
end



