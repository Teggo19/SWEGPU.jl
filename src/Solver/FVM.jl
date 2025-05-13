using KernelAbstractions
using StaticArrays

function F(h::T, hu::T, hv::T) where {T<:Real}

    return hu, hu^2/h + 9.81f0*0.5f0*h^2, hu*hv/h
end



function compute_eigenvalues_F(h, hu, hv)
    u = hu/h
    #println("U : ", U)
    return u + sqrt(9.81f0*h), u - sqrt(9.81f0*h) #, u Dropping the last eigenvalue because it is never the largest or smallest one
end



function central_upwind_flux_kurganov(h1, hu1, hv1, h2, hu2, hv2, flux, compute_eigenvalues)
    # This could potentially be G and not F
    hf1, huf1, hvf1 = flux(h1, hu1, hv1)
    hf2, huf2, hvf2 = flux(h2, hu2, hv2)

    ev1_pos, ev1_neg = compute_eigenvalues(h1, hu1, hv1)
    ev2_pos, ev2_neg = compute_eigenvalues(h2, hu2, hv2)

    aplus = max(ev1_pos, ev2_pos, 0.f0)
    aminus = min(ev1_neg, ev2_neg, 0.f0)
    #println("aplus : ", aplus, "  aminus : ", aminus)
    F1 = (aplus * hf1 - aminus * hf2)/(aplus - aminus) + ((aplus * aminus) / (aplus - aminus)) * (h2 - h1)
    F2 = (aplus * huf1 - aminus * huf2)/(aplus - aminus) + ((aplus * aminus) / (aplus - aminus)) * (hu2 - hu1)
    F3 = (aplus * hvf1 - aminus * hvf2)/(aplus - aminus) + ((aplus * aminus) / (aplus - aminus)) * (hv2 - hv1)
    return F1, F2, F3, max(abs(aplus), abs(aminus))
end

function rotate_u(hu, hv, n1, n2)
    return hu*n1 - hv*n2, hu*n2 + hv*n1
end

function rotate_u_back(hu, hv, n1, n2)
    return hu*n1 + hv*n2, -hu*n2+hv*n1
end

@kernel function update_fluxes!(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc)
    i = @index(Global)

    # Checking if the edge is a boundary edge
    cell1 = edge_cell_matrix[i, 1]
    h1, hu1, hv1 = U[cell1, 1], U[cell1, 2], U[cell1, 3]

    n1, n2 = normal_matrix[i, 1], normal_matrix[i, 2]
    hu1_rot, hv1_rot = rotate_u(hu1, hv1, n1, n2)
    
    # Add reconstruction here

    if (edge_cell_matrix[i, 2] == 0) 
        f1, f2_rot, f3_rot, lambda = bc(h1, hu1_rot, hv1_rot, F, compute_eigenvalues_F)
        cell2 = edge_cell_matrix[i, 1]
    else
        cell2 = edge_cell_matrix[i, 2]
        h2, hu2, hv2 = U[cell2, 1], U[cell2, 2], U[cell2, 3]
        
        hu2_rot, hv2_rot = rotate_u(hu2, hv2, n1, n2)

        f1, f2_rot, f3_rot, lambda = central_upwind_flux_kurganov(h1, hu1_rot, hv1_rot, h2, hu2_rot, hv2_rot, F, compute_eigenvalues_F)
        
    end
    f2, f3 = rotate_u_back(f2_rot, f3_rot, n1, n2)

    # Should add gradient of the topography here

    # Updating the fluxes
    fluxes[i, 1] = f1*edge_lengths[i]
    fluxes[i, 2] = f2*edge_lengths[i]
    fluxes[i, 3] = f3*edge_lengths[i]
    
    
    # Get the diameter for each of the cells
    diameter_1 = diameters[cell1]
    # could add check if cell1=cell2
    diameter_2 = diameters[cell2]

    # Update the max_dt_array according to the CFL condition
    if lambda != 0
        if  diameter_1/lambda < max_dt_array[cell1]
            max_dt_array[cell1] = diameter_1/lambda
        end
        if diameter_2/lambda < max_dt_array[cell2]
            max_dt_array[cell2] = diameter_2/lambda
        end
    end
end

@kernel function update_values!(U, fluxes, cell_edge_matrix, edge_cell_matrix, cell_areas, dt, max_dt_array, update_dt)
    i = @index(Global)
        
    if update_dt
        max_dt_array[i] = 1.f0
    end
    #edges = [cell_edge_matrix[i, 1], cell_edge_matrix[i, 2], cell_edge_matrix[i, 3]]
    
    for j in 1:3
        edge = cell_edge_matrix[i, j]
        if edge_cell_matrix[edge, 1] == i
            #println("Updating cell $i with edge $edge. Value: $(fluxes[edge, :])")
            a = dt/cell_areas[i]
            U[i, 1] -= a*fluxes[edge, 1]
            U[i, 2] -= a*fluxes[edge, 2]
            U[i, 3] -= a*fluxes[edge, 3]
            
        elseif edge_cell_matrix[edge, 2] == i
            a = dt/cell_areas[i]
            U[i, 1] += a*fluxes[edge, 1]
            U[i, 2] += a*fluxes[edge, 2]
            U[i, 3] += a*fluxes[edge, 3]
        #= else
            println("Error: Mismanaged edge-cell matrixes") =#
        end
        
    end
end

@kernel function update_fluxes_with_reconstruction!(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, recon_gradient, edge_coordinates, centroids)
    i = @index(Global)

    # Checking if the edge is a boundary edge
    cell1 = edge_cell_matrix[i, 1]
    edge_coord = 0.5*(edge_coordinates[i, 1, :] + edge_coordinates[i, 2, :]) - centroids[cell1, 1:2]
    (h1, hu1, hv1) = (U[cell1, 1], U[cell1, 2], U[cell1, 3]) .+ recon_gradient[cell1, :, 1] * edge_coord[1] .+ recon_gradient[cell1, :, 2] * edge_coord[2]


    n1, n2 = normal_matrix[i, 1], normal_matrix[i, 2]
    hu1_rot, hv1_rot = rotate_u(hu1, hv1, n1, n2)
    
    # Add reconstruction here

    if (edge_cell_matrix[i, 2] == 0) 
        f1, f2_rot, f3_rot, lambda = bc(h1, hu1_rot, hv1_rot, F, compute_eigenvalues_F)
        cell2 = edge_cell_matrix[i, 1]
    else
        cell2 = edge_cell_matrix[i, 2]
        (h2, hu2, hv2) = (U[cell2, 1], U[cell2, 2], U[cell2, 3]) .+ recon_gradient[cell2, :, 1] * edge_coord[1] .+ recon_gradient[cell2, :, 2] * edge_coord[2]
        
        hu2_rot, hv2_rot = rotate_u(hu2, hv2, n1, n2)

        f1, f2_rot, f3_rot, lambda = central_upwind_flux_kurganov(h1, hu1_rot, hv1_rot, h2, hu2_rot, hv2_rot, F, compute_eigenvalues_F)
        
    end
    f2, f3 = rotate_u_back(f2_rot, f3_rot, n1, n2)

    # Should add gradient of the topography here

    # Updating the fluxes
    fluxes[i, 1] = f1*edge_lengths[i]
    fluxes[i, 2] = f2*edge_lengths[i]
    fluxes[i, 3] = f3*edge_lengths[i]
    
    
    # Get the diameter for each of the cells
    diameter_1 = diameters[cell1]
    # could add check if cell1=cell2
    diameter_2 = diameters[cell2]

    # Update the max_dt_array according to the CFL condition
    if lambda != 0
        if  diameter_1/lambda < max_dt_array[cell1]
            max_dt_array[cell1] = diameter_1/lambda
        end
        if diameter_2/lambda < max_dt_array[cell2]
            max_dt_array[cell2] = diameter_2/lambda
        end
    end
end

@kernel function avg_kernel!(val1, val2)
    i = @index(Global)
    res = (val1[i] + val2[i])/2
    val1[i] = res
    val2[i] = res
end