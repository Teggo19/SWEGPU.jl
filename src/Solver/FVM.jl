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

    return hu*n1 + hv*n2, -hu*n2 + hv*n1
end

function rotate_u_back(hu, hv, n1, n2)

    return hu*n1 - hv*n2, hu*n2 + hv*n1
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
    diameter = min(diameter_1, diameter_2)

    
    # Update the max_dt_array according to the CFL condition
    if lambda != 0
        max_dt_array[i] = diameter/lambda
        
        #=
        if  diameter_1/lambda < max_dt_array[cell1]
            max_dt_array[cell1] = diameter_1/lambda
        end
        if diameter_2/lambda < max_dt_array[cell2]
            max_dt_array[cell2] = diameter_2/lambda
        end
        =#
    end
end

@kernel function update_values!(U, fluxes, cell_edge_matrix, edge_cell_matrix, cell_areas, dt, max_dt_array, update_dt)
    i = @index(Global)
    
    if update_dt
        max_dt_array[i] = 1.f0
    end
    #edges = [cell_edge_matrix[i, 1], cell_edge_matrix[i, 2], cell_edge_matrix[i, 3]]
    
    a = dt/cell_areas[i]
    val1 = U[i, 1]
    val2 = U[i, 2]
    val3 = U[i, 3]
    for j in 1:3
        edge = cell_edge_matrix[i, j]
        if edge_cell_matrix[edge, 1] == i
            #println("Updating cell $i with edge $edge. Value: $(fluxes[edge, :])")
            
            val1 -= a*fluxes[edge, 1]
            val2 -= a*fluxes[edge, 2]
            val3 -= a*fluxes[edge, 3]
            
        elseif edge_cell_matrix[edge, 2] == i
            
            val1 += a*fluxes[edge, 1]
            val2 += a*fluxes[edge, 2]
            val3 += a*fluxes[edge, 3]
        #= else
            println("Error: Mismanaged edge-cell matrixes") =#
        end
        
    end
    U[i, 1] = val1
    U[i, 2] = val2
    U[i, 3] = val3
end

@kernel function update_values_with_recon!(U, fluxes, cell_edge_matrix, edge_cell_matrix, cell_areas, dt, max_dt_array, update_dt, recon_gradient, normal_matrix, edge_lengths, edge_coordinates, centroids, cell_grads)
    i = @index(Global)
    
    if update_dt
        max_dt_array[i] = 1.f0
    end
    #edges = [cell_edge_matrix[i, 1], cell_edge_matrix[i, 2], cell_edge_matrix[i, 3]]
    
    a = dt/cell_areas[i]
    val1 = U[i, 1]
    val2 = U[i, 2]
    val3 = U[i, 3]

    h = U[i, 1]
    for j in 1:3
        edge = cell_edge_matrix[i, j]

        # Bottom topography gradient
        n1, n2 = normal_matrix[edge, 1], normal_matrix[edge, 2]
        l = edge_lengths[edge]

        h_edgeval = h + (recon_gradient[i, 1, 1]-cell_grads[i, 1]) * (edge_coordinates[edge, 1] - centroids[i, 1]) + (recon_gradient[i, 1, 2]-cell_grads[i, 2]) * (edge_coordinates[edge, 2] - centroids[i, 2])

        if edge_cell_matrix[edge, 1] == i

            val1 -= a*fluxes[edge, 1]
            val2 -= a*fluxes[edge, 2]
            val3 -= a*fluxes[edge, 3]

            val2 += a*0.5f0*9.81f0*l*((h_edgeval)^2)*n1
            val3 += a*0.5f0*9.81f0*l*((h_edgeval)^2)*n2
            
        elseif edge_cell_matrix[edge, 2] == i

            val1 += a*fluxes[edge, 1]
            val2 += a*fluxes[edge, 2]
            val3 += a*fluxes[edge, 3]

            val2 -= a*0.5f0*9.81f0*l*((h_edgeval)^2)*n1
            val3 -= a*0.5f0*9.81f0*l*((h_edgeval)^2)*n2
        #= else
            println("Error: Mismanaged edge-cell matrixes") =#
        end

        #println("Cell $i, h: $h, edge: $edge, h_edgeval: $(h_edgeval), length: $l, recon_gradient: $(recon_gradient[i, :, :]), cell_grads: $(cell_grads[i, :]), edge_coord: $(edge_coordinates[edge, :] - centroids[i, :]) \n")

        # Balance terms from the topography gradient
        
    end
    #println("\n\n")
    
    val2 -= dt*9.81f0*recon_gradient[i, 1, 1]*h
    val3 -= dt*9.81f0*recon_gradient[i, 1, 2]*h

    U[i, 1] = val1
    U[i, 2] = val2
    U[i, 3] = val3
end

@kernel function update_fluxes_with_reconstruction!(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters, bc, recon_gradient, edge_coordinates, centroids, cell_grads)
    i = @index(Global)

    # Checking if the edge is a boundary edge
    cell1 = edge_cell_matrix[i, 1]
    
    edge_coord = SVector(edge_coordinates[i, 1] - centroids[cell1, 1], edge_coordinates[i, 2] - centroids[cell1, 2])
    #SVector(0.5*(edge_coordinates[i, 1, 1] + edge_coordinates[i, 2, 1]) - centroids[cell1, 1], 0.5*(edge_coordinates[i, 1, 2] + edge_coordinates[i, 2, 2]) - centroids[cell1, 2])
    h1 = U[cell1, 1] + (recon_gradient[cell1, 1, 1] - cell_grads[cell1, 1]) * edge_coord[1] + (recon_gradient[cell1, 1, 2]-cell_grads[cell1, 2]) * edge_coord[2]
    hu1 = U[cell1, 2] + recon_gradient[cell1, 2, 1] * edge_coord[1] + recon_gradient[cell1, 2, 2] * edge_coord[2]
    hv1 = U[cell1, 3] + recon_gradient[cell1, 3, 1] * edge_coord[1] + recon_gradient[cell1, 3, 2] * edge_coord[2]


    n1, n2 = normal_matrix[i, 1], normal_matrix[i, 2]
    hu1_rot, hv1_rot = rotate_u(hu1, hv1, n1, n2)

    if (edge_cell_matrix[i, 2] == 0) 
        f1, f2_rot, f3_rot, lambda = bc(h1, hu1_rot, hv1_rot, F, compute_eigenvalues_F)
        cell2 = edge_cell_matrix[i, 1]
        # edge_coord2 = edge_coord
    else
        cell2 = edge_cell_matrix[i, 2]
        edge_coord2 = SVector(edge_coordinates[i, 1] - centroids[cell2, 1], edge_coordinates[i, 2] - centroids[cell2, 2])
        h2 = U[cell2, 1] + (recon_gradient[cell2, 1, 1]-cell_grads[cell2, 1]) * edge_coord2[1] + (recon_gradient[cell2, 1, 2]-cell_grads[cell2, 2]) * edge_coord2[2]
        hu2 = U[cell2, 2] + recon_gradient[cell2, 2, 1] * edge_coord2[1] + recon_gradient[cell2, 2, 2] * edge_coord2[2]
        hv2 = U[cell2, 3] + recon_gradient[cell2, 3, 1] * edge_coord2[1] + recon_gradient[cell2, 3, 2] * edge_coord2[2]

        
        hu2_rot, hv2_rot = rotate_u(hu2, hv2, n1, n2)

        f1, f2_rot, f3_rot, lambda = central_upwind_flux_kurganov(h1, hu1_rot, hv1_rot, h2, hu2_rot, hv2_rot, F, compute_eigenvalues_F)
        
        
    end
    

    #if cell1 == 9 || cell2 == 9
    #    println("Cell1 = $cell1, Cell2 = $cell2, Edge $i: h1 = $h1, h2 = $h2 \nrecon1 = ($(recon_gradient[cell1, 1, 1]), $(recon_gradient[cell1, 1, 2])), recon2 = ($(recon_gradient[cell2, 1, 1]) $(recon_gradient[cell2, 1, 2])) \nedge_coord = $edge_coord, edge_coord2 = $edge_coord2 \n ")
    #end
    
    
    f2, f3 = rotate_u_back(f2_rot, f3_rot, n1, n2)

    # Updating the fluxes
    fluxes[i, 1] = f1*edge_lengths[i]
    fluxes[i, 2] = f2*edge_lengths[i]
    fluxes[i, 3] = f3*edge_lengths[i]


    if i == 7 || i == 18
        #=println("Cell1 = $cell1, h1 = $h1, hu1 = $hu1, hv1 = $hv1, U1 : $(U[cell1, :])\n
        Cell2 = $cell2, h2 = $h2, hu2 = $hu2, hv2 = $hv2, U2 : $(U[cell2, :])\n
        f1 = $f1, f2 = $f2, f3=$f3\n
        edge_coord = $edge_coord, edge_coord2 = $edge_coord2 \n
        recon1 = $(recon_gradient[cell1, 2, :])\n
        recon2 = $(recon_gradient[cell2, 3, :])\n
        n = ($n1, $n2) \n")=#
    end
    
    
    # Get the diameter for each of the cells
    diameter_1 = diameters[cell1]
    #diameter_1 = sqrt(edge_coord[1]^2 + edge_coord[2]^2)
    # could add check if cell1=cell2
    diameter_2 = diameters[cell2]
    #diameter_2 = sqrt(edge_coord2[1]^2 + edge_coord2[2]^2)
    diameter = min(diameter_1, diameter_2)
    
    #TODO: Fix the CFL to the same as in the paper
    # Update the max_dt_array according to the CFL condition
    if lambda != 0
        max_dt_array[i] = diameter/lambda
    end
    
end

@kernel function avg_kernel!(val1, val2)
    i = @index(Global)
    for j in 1:3
        res = (val1[i, j] + val2[i, j])/2
        val1[i, j] = res
        val2[i, j] = res
    end
end