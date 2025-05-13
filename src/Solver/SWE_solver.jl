using KernelAbstractions
using ProgressMeter
using CUDA

include("../structs.jl")
include("FVM.jl")
include("bc.jl")
include("reconstruction.jl")

function SWE_solver(cells, edges, T, initial; backend="CPU", bc=neumannBC, reconstruction=0)
    # TODO: Dynamic type allocation to allow for Automatic Differentiation
    
    spaceType = eltype(cells[1].centroid)
    indType = eltype(cells[1].edges)
    # Set initial conditions
    n_edges = length(edges)
    n_cells = length(cells)
    
    t = 0

    U = deepcopy(initial)
    
    # Holds information of what cells each edge is connected to
    edge_cell_matrix = make_edge_cell_matrix(edges)
    centroids = hcat([cell.centroid for cell in cells]...)'
    areas = hcat([cell.area for cell in cells]...)'
    diameters = hcat([cell.diameter for cell in cells]...)'
    edge_lengths = hcat([edge.length for edge in edges]...)'

    edge_coordinates = make_edge_coordinates_array(edges)

    # Holds information of what edges each cell is connected to
    cell_edge_matrix = make_cell_edge_matrix(cells)
    

    fluxes = zeros(spaceType, n_edges, 3)
    normal_matrix = make_normal_matrix(edges)

    max_dt_array = ones(spaceType, n_cells)

    if reconstruction == 1
        recon_gradient = zeros(spaceType, n_cells, 3, 2)
        U2 = deepcopy(U)
    end

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
        centroids = CuArray(centroids)
        edge_coordinates = CuArray(edge_coordinates)

        if reconstruction == 1
            recon_gradient = CuArray(recon_gradient)
            U2 = CuArray(U2)
        end


        dev = get_backend(U)
    
    else
        dev = CPU()
    end

    # Loop over time
    p = Progress(Int64(ceil(T*3840)); dt=0.1)
    
    CFL = 0.25
    while t < T
        if reconstruction == 1
            update_reconstruction!(dev, 64)(U, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates, ndrange=n_cells)
            #return recon_gradient
            update_fluxes_with_reconstruction!(dev, 64)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, recon_gradient, edge_coordinates, centroids, ndrange=n_edges)
        
        # Loop over edges
        else
            update_fluxes!(dev, 64)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, ndrange=n_edges)
        end
        KernelAbstractions.synchronize(dev)
        #println("fluxes : ", fluxes)
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
        if reconstruction == 1
            update_values!(dev, 64)(U2, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, max_dt_array, false, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)
            update_reconstruction!(dev, 64)(U2, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)
            update_values!(dev, 64)(U2, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, max_dt_array, true, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)
            avg_kernel!(dev, 64)(U, U2, ndrange=n_cells)
        else
            update_values!(dev, 64)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, max_dt_array, true, ndrange=n_cells)
        end
        #update_values!(dev, 64)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, ndrange=n_cells)
        KernelAbstractions.synchronize(dev)


        update!(p, Int64(ceil(t*3840)))
    end
    finish!(p)
    if backend == "CUDA"
        U = collect(U)
    end
    return U
end



