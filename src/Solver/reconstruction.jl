using KernelAbstractions

@kernel function update_reconstruction!(U, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates)
    i = @index(Global)

    c = centroids[i, :]
    u = U[i, :]

    neighbours = find_neighbours(cell_edge_matrix, edge_cell_matrix, i)
    if length(neighbours) != 3
        # boundary edge
        for j in 1:3
            recon_gradient[i, j, 1:2] .= [0.0, 0.0]
        end
    else
        u1 = U[neighbours[1], :]
        u2 = U[neighbours[2], :]
        u3 = U[neighbours[3], :]
        c1 = centroids[neighbours[1], :]
        c2 = centroids[neighbours[2], :]
        c3 = centroids[neighbours[3], :]
        # Compute the gradients
        
        for j in 1:3
            val = u[j]
            val1 = u1[j]
            val2 = u2[j]
            val3 = u3[j]
            
            dx1 = c1[1] - c[1]
            dx2 = c2[1] - c[1]
            dx3 = c3[1] - c[1]
            
            dy1 = c1[2] - c[2]
            dy2 = c2[2] - c[2]
            dy3 = c3[2] - c[2]
            
            du1 = val1 - val
            du2 = val2 - val
            du3 = val3 - val
            
            # Compute the gradient
            grad12 = ((du1 * dy2 - du2 * dy1) / (dx1 * dy2 - dx2 * dy1), 
                    (du1 * dx2 - du2 * dx1) / (dy1 * dx2 - dy2 * dx1))
            
            grad13 = ((du1 * dy3 - du3 * dy1) / (dx1 * dy3 - dx3 * dy1),
                    (du1 * dx3 - du3 * dx1) / (dy1 * dx3 - dy3 * dx1))

            grad23 = ((du2 * dy3 - du3 * dy2) / (dx2 * dy3 - dx3 * dy2),
                    (du2 * dx3 - du3 * dx2) / (dy2 * dx3 - dy3 * dx2))

            abs_grad12 = sqrt(grad12[1]^2 + grad12[2]^2)
            abs_grad13 = sqrt(grad13[1]^2 + grad13[2]^2)
            abs_grad23 = sqrt(grad23[1]^2 + grad23[2]^2)


            grad_written = false
            if abs_grad12 <= abs_grad13 && abs_grad12 <= abs_grad23
                if check_gradient(grad12, val, val3, c, edge_coordinates[cell_edge_matrix[i, 3], 1, :], edge_coordinates[cell_edge_matrix[i, 3], 2, :])
                    recon_gradient[i, j, 1:2] .= grad12
                    grad_written = true
                else
                    recon_gradient[i, j, 1:2] .= [0.0, 0.0]
                end
            end
            if !grad_written && abs_grad13 <= abs_grad23 && abs_grad13 <= abs_grad12
                if check_gradient(grad13, val, val2, c, edge_coordinates[cell_edge_matrix[i, 2], 1, :], edge_coordinates[cell_edge_matrix[i, 2], 2, :])
                    recon_gradient[i, j, 1:2] .= grad13
                    grad_written = true
                else
                    recon_gradient[i, j, 1:2] .= [0.0, 0.0]
                end
            end
            if !grad_written && abs_grad23 <= abs_grad12 && abs_grad23 <= abs_grad13
                if check_gradient(grad12, val, val1, c, edge_coordinates[cell_edge_matrix[i, 1],1, :], edge_coordinates[cell_edge_matrix[i, 1],2, :])
                    recon_gradient[i, j, 1:2] .= grad23
                    grad_written = true
                else
                    recon_gradient[i, j, 1:2] .= [0.0, 0.0]
                end
            end
            
        end
        
    end
end


function find_neighbours(cell_edge_matrix, edge_cell_matrix, i)
    neighbours = zeros(eltype(cell_edge_matrix[1, :]), 0)
    for j in 1:3
        edge = cell_edge_matrix[i, j]
        if edge_cell_matrix[edge, 1] == i
            if edge_cell_matrix[edge, 2] != 0
                push!(neighbours, edge_cell_matrix[edge, 2])
            end
        else
            push!(neighbours, edge_cell_matrix[edge, 1])
        end
    end
    return neighbours
end

function check_gradient(gradient, val, val2, c, pt1, pt2)
    edge_centre = (pt1 + pt2) / 2
    diff = edge_centre - c[1:2]
    
    edge_val = val + gradient[1] * diff[1] + gradient[2] * diff[2]
    if edge_val >= val && edge_val <= val2
        return true
    elseif edge_val <= val && edge_val >= val2
        return true
    else
        return false
    end
end