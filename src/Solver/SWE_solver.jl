using KernelAbstractions
using ProgressMeter
using CUDA


include("../structs.jl")
include("FVM.jl")
include("bc.jl")
include("reconstruction.jl")

function SWE_solver(cells, edges, T, initial; backend="CPU", bc=neumannBC, time_stepper=1, return_runtime=false, CFL=0.5f0, limiter=0)
    # TODO: Dynamic type allocation to allow for Automatic Differentiation
    blocksize = 256
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
    cell_grads = hcat([cell.grad for cell in cells]...)'
    areas = hcat([cell.area for cell in cells]...)'
    #diameters = hcat([cell.diameter for cell in cells]...)'
    edge_lengths = hcat([edge.length for edge in edges]...)'
    edge_coordinates = make_edge_center_matrix(edges)

    # Holds information of what edges each cell is connected to
    cell_edge_matrix = make_cell_edge_matrix(cells)
    
    diameters = make_diameter_array(cells, edges)

    fluxes = zeros(spaceType, n_edges, 3)
    normal_matrix = make_normal_matrix(edges)

    max_dt_array = ones(spaceType, n_edges)

    recon_gradient = zeros(spaceType, n_cells, 3, 2)
    if time_stepper == 1
        
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
        cell_grads = CuArray(cell_grads)
        
        edge_coordinates = CuArray(edge_coordinates)
        recon_gradient = CuArray(recon_gradient)

        if time_stepper == 1     
            U2 = CuArray(U2)
        end


        dev = get_backend(U)
    
    else
        dev = get_backend(U)
    end

    # Loop over time
    p = Progress(Int64(ceil(T*3840)); dt=0.1)
    
    CFL = convert(spaceType, CFL)

    if return_runtime
        start_time = time()
        n_timesteps = 0
    end
    while t < T
        if time_stepper == 1

            update_reconstruction!(dev, blocksize)(U, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates, limiter, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)

            update_fluxes_with_reconstruction!(dev, blocksize)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, recon_gradient, edge_coordinates, centroids, cell_grads, ndrange=n_edges)
            KernelAbstractions.synchronize(dev)

            #update_fluxes!(dev, blocksize)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, ndrange=n_edges)
            #KernelAbstractions.synchronize(dev)

            dt = T - t
            if backend =="CUDA"
                max_dt = CUDA.minimum(max_dt_array)
                KernelAbstractions.synchronize(dev)
            else
                max_dt = minimum(max_dt_array)
            end
            dt = min(dt, CFL*max_dt)
            t += dt


            update_values_with_recon!(dev, blocksize)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, max_dt_array, false, recon_gradient, normal_matrix, edge_lengths, edge_coordinates, centroids, cell_grads, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)

            update_reconstruction!(dev, blocksize)(U, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates, limiter, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)

            update_fluxes_with_reconstruction!(dev, blocksize)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, recon_gradient, edge_coordinates, centroids, cell_grads, ndrange=n_edges)
            KernelAbstractions.synchronize(dev)

            update_values_with_recon!(dev, blocksize)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, max_dt_array, true, recon_gradient, normal_matrix, edge_lengths, edge_coordinates, centroids, cell_grads, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)
  

            avg_kernel!(dev, blocksize)(U, U2, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)

        # Loop over edges
        else
            update_reconstruction!(dev, blocksize)(U, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates, limiter, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)
            
            update_fluxes_with_reconstruction!(dev, blocksize)(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, recon_gradient, edge_coordinates, centroids, cell_grads, ndrange=n_edges)
            KernelAbstractions.synchronize(dev)

            dt = T - t
            if backend =="CUDA"
                max_dt = CUDA.minimum(max_dt_array)
                KernelAbstractions.synchronize(dev)
            else
                max_dt = minimum(max_dt_array)
            end
            dt = min(dt, CFL*max_dt)
            t += dt

            # Loop over cells
            update_values_with_recon!(dev, blocksize)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, max_dt_array, false, recon_gradient, normal_matrix, edge_lengths, edge_coordinates, centroids, cell_grads, ndrange=n_cells)
            KernelAbstractions.synchronize(dev)

        end
        #update_values!(dev, 64)(U, fluxes, cell_edge_matrix, edge_cell_matrix, areas, dt, ndrange=n_cells)

        #println("dt: $dt, t: $t")
        update!(p, Int64(ceil(t*3840)))

        if return_runtime
            n_timesteps += 1
        end
    end
    finish!(p)
    if backend == "CUDA"
        U = collect(U)
    end
    if return_runtime
        runtime = time() - start_time
        println("Runtime: $runtime seconds")
        return U, runtime, n_timesteps
    end
    return U
end




