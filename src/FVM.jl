using KernelAbstractions
g = 9.81f0

function F(U)
    # TODO: change type to the same as U
    res = Vector{Float32}(undef, 3)
    res[1] = U[2]
    res[2] = U[2]^2/U[1] + 0.5*U[1]^2*g
    res[3] = U[2]*U[3]/U[1]
    return res 
end

function G(U)
    # TODO: change type to the same as U
    res = Vector{Float32}(undef, 3)
    res[1] = U[3]
    res[2] = U[2]*U[3]/U[1]
    res[3] = U[3]^2/U[1] + 0.5*U[1]^2*g
    return res
end

function compute_eigenvalues_F(U)
    u = U[2]/U[1]
    #println("U : ", U)
    return [u + sqrt(g*U[1]), u - sqrt(g*U[1]), u]
end

function compute_eigenvalues_G(U)
    v = U[3]/U[1]
    return [v + sqrt(g*U[1]), v - sqrt(g*U[1]), v]
end


function central_upwind_flux_kurganov(U1, U2, flux, compute_eigenvalues)
    # This could potentially be G and not F
    f1 = flux(U1)
    f2 = flux(U2)

    ev1 = compute_eigenvalues(U1)
    ev2 = compute_eigenvalues(U2)

    aplus = max(ev1[1], ev2[1], 0.0)
    aminus = min(ev1[2], ev2[2], 0.0)
    #println("aplus : ", aplus, "  aminus : ", aminus)
    F = (aplus .* f1 - aminus .* f2)./(aplus - aminus) + ((aplus .* aminus) ./ (aplus - aminus)) .* (U2 - U1)
    return F, max(abs(aplus), abs(aminus))
end

@kernel function update_fluxes!(fluxes, U, edge_cell_matrix, normal_matrix, edge_lengths, max_dt_array, diameters)
    i = @index(Global)

    # Checking if the edge is a boundary edge
    cell1 = edge_cell_matrix[i, 1]
    if (edge_cell_matrix[i, 2] == 0) 
        cell2 = edge_cell_matrix[i, 1]
    else
        cell2 = edge_cell_matrix[i, 2]
    end
    U1 = U[cell1, :]
    U2 = U[cell2, :]
    n = normal_matrix[i, :]

    fluxF, lambdaF = central_upwind_flux_kurganov(U1, U2, F, compute_eigenvalues_F)
    fluxG, lambdaG = central_upwind_flux_kurganov(U1, U2, G, compute_eigenvalues_G)
    #println("edge : $i, FluxF : $fluxF, fluxG : $fluxG, n : $n, edge_length : $(edge_lengths[i])")
    fluxes[i, :] = fluxF.*(n[1]*edge_lengths[i])+fluxG.*(n[2]*edge_lengths[i])

    # Get the diameter for each of the cells
    diameter_1 = diameters[cell1]
    # could add check if cell1=cell2
    diameter_2 = diameters[cell2]

    max_lambda = max(lambdaF, lambdaG)
    #println("lambdaF : ", lambdaF, "  lambdaG : ", lambdaG)
    #println("max_lambda :", max_lambda)
    if max_lambda != 0
        if  diameter_1/max_lambda < max_dt_array[cell1]
            max_dt_array[cell1] = diameter_1/max_lambda
        end
        if diameter_2/max_lambda < max_dt_array[cell2]
            max_dt_array[cell2] = diameter_2/max_lambda
        end
    end
            
    
end

@kernel function update_values!(U, fluxes, cell_edge_matrix, edge_cell_matrix, cell_areas, dt)
    i = @index(Global)

    edges = cell_edge_matrix[i, :]

    for edge in edges
        if edge_cell_matrix[edge, 1] == i
            #println("Updating cell $i with edge $edge. Value: $(fluxes[edge, :])")
            U[i, :] -= dt/cell_areas[i]*fluxes[edge, :]
        end
        if !(edge_cell_matrix[edge, 2] == 0)
            
            if edge_cell_matrix[edge, 2] == i
                U[i, :] += dt/cell_areas[i]*fluxes[edge, :]
            #= else
                println("Error: Mismanaged edge-cell matrixes") =#
            end
        end
    end
end