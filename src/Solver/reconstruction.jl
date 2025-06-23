using KernelAbstractions
using StaticArrays

@kernel function update_reconstruction!(U, recon_gradient, centroids, cell_edge_matrix, edge_cell_matrix, edge_coordinates, limiter; bc=0)
    i = @index(Global)
    spaceType = eltype(U)
    c = SVector(centroids[i, 1], centroids[i, 2])
    # add the +c[3] to account for the bottom topography
    u = SVector(U[i, 1] + centroids[i, 3], U[i, 2], U[i, 3])
    
    neighbours = find_neighbours(cell_edge_matrix, edge_cell_matrix, i)
    if neighbours[1] == 0 || neighbours[2] == 0 || neighbours[3] == 0
        # If there are no neighbours, we cannot compute the gradient, so we set it to zero
        for j in 1:3
            recon_gradient[i, j, 1] = 0.f0
            recon_gradient[i, j, 2] = 0.f0
        end
    else
    if neighbours[1] == 0
        # If the neighbour is 0, it means there is no neighbour in that direction, we need to construct a ghost cell with boundary conditions
        if bc == 0 # Neumann boundary condition
            u1 = SVector(u[1], u[2], u[3])
            
            edge_distance = SVector(edge_coordinates[cell_edge_matrix[i, 1], 1] - c[1], edge_coordinates[cell_edge_matrix[i, 1], 2] - c[2])
            c1 = SVector(c[1] + 2 * edge_distance[1], c[2] + 2 * edge_distance[2])
        end
    else
        u1 = SVector(U[neighbours[1], 1] + centroids[neighbours[1], 3], U[neighbours[1], 2], U[neighbours[1], 3])
        c1 = SVector(centroids[neighbours[1], 1], centroids[neighbours[1], 2])
    end
    if neighbours[2] == 0
        if bc == 0 # Neumann boundary condition
            u2 = SVector(u[1], u[2], u[3])
            
            edge_distance = SVector(edge_coordinates[cell_edge_matrix[i, 2], 1] - c[1], edge_coordinates[cell_edge_matrix[i, 2], 2] - c[2])
            c2 = SVector(c[1] + 2 * edge_distance[1], c[2] + 2 * edge_distance[2])
        end
    else
        u2 = SVector(U[neighbours[2], 1] + centroids[neighbours[2], 3], U[neighbours[2], 2], U[neighbours[2], 3])
        c2 = SVector(centroids[neighbours[2], 1], centroids[neighbours[2], 2])
    end
    if neighbours[3] == 0
        if bc == 0 # Neumann boundary condition
            u3 = SVector(u[1], u[2], u[3])
            
            edge_distance = SVector(edge_coordinates[cell_edge_matrix[i, 3], 1] - c[1], edge_coordinates[cell_edge_matrix[i, 3], 2] - c[2])
            c3 = SVector(c[1] + 2 * edge_distance[1], c[2] + 2 * edge_distance[2])
        end
    else
        u3 = SVector(U[neighbours[3], 1] + centroids[neighbours[3], 3], U[neighbours[3], 2], U[neighbours[3], 3])
        c3 = SVector(centroids[neighbours[3], 1], centroids[neighbours[3], 2])
    end
    

    # Compute the gradients
    dx1 = c1[1] - c[1]
    dx2 = c2[1] - c[1]
    dx3 = c3[1] - c[1]
    
    dy1 = c1[2] - c[2]
    dy2 = c2[2] - c[2]
    dy3 = c3[2] - c[2]
        
    for j in 1:3
        
        val = u[j]
        vals = SVector(u1[j], u2[j], u3[j])
        
        
        
        du1 = vals[1] - val
        du2 = vals[2] - val
        du3 = vals[3] - val
        
        # Compute the gradient
        #=grad12 = SVector((du1 * dy2 - du2 * dy1) / (dx1 * dy2 - dx2 * dy1), 
                (du1 * dx2 - du2 * dx1) / (dy1 * dx2 - dy2 * dx1))
        
        grad13 = SVector((du1 * dy3 - du3 * dy1) / (dx1 * dy3 - dx3 * dy1),
                (du1 * dx3 - du3 * dx1) / (dy1 * dx3 - dy3 * dx1))

        grad23 = SVector((du2 * dy3 - du3 * dy2) / (dx2 * dy3 - dx3 * dy2),
                (du2 * dx3 - du3 * dx2) / (dy2 * dx3 - dy3 * dx2))
        if any(isnan, grad23)
            #println("du2: $du2, du3: $du3, dy2: $dy2, dy3: $dy3, dx2: $dx2, dx3: $dx3")
        end=#
        
        grads = SMatrix{3, 2}((du1 * dy2 - du2 * dy1) / (dx1 * dy2 - dx2 * dy1), (du1 * dy3 - du3 * dy1) / (dx1 * dy3 - dx3 * dy1), (du2 * dy3 - du3 * dy2) / (dx2 * dy3 - dx3 * dy2),
                                (du1 * dx2 - du2 * dx1) / (dy1 * dx2 - dy2 * dx1), (du1 * dx3 - du3 * dx1) / (dy1 * dx3 - dy3 * dx1), (du2 * dx3 - du3 * dx2) / (dy2 * dx3 - dy3 * dx2))

                                
                                
                                
        if limiter == 0
            abs_grads = SVector(grads[1, 1]^2 + grads[1, 2]^2, grads[2, 1]^2 + grads[2, 2]^2, grads[3, 1]^2 + grads[3, 2]^2)

            grad_ind = argmin(abs_grads)

            grad = grads[grad_ind, :]
            grad_test = true
            for k in 1:3
                edge_coord = SVector(edge_coordinates[cell_edge_matrix[i, k], 1], edge_coordinates[cell_edge_matrix[i, k], 2])
                edge_val = val + grad[1] * (edge_coord[1] - c[1]) + grad[2] * (edge_coord[2] - c[2])
                if (edge_val > val && edge_val > vals[k]) || (edge_val < val && edge_val < vals[k])
                    grad_test = false
                    if (i == 3 && j == 3)
                        #println("Gradient test failed for cell $i, edge $k, grad = $grad, edge_val = $edge_val, val = $val, vals[$k] = $(vals[k]), edge_coord = $edge_coord")
                    end
                    break
                end
            end
            if !grad_test
                grad = SVector(0.f0, 0.f0)
            end
            if  (i == 9 && j == 2) || (i == 3 && j == 3)
                #println("Cell $i, grads = $(grads), val = $val, vals = $vals, grad = $grad, grad_test = $grad_test")
            end

        elseif limiter == 1
            g1 = 0.f0
            g2 = 0.f0
            if (sign(grads[1, 1]) == sign(grads[2, 1]) && sign(grads[1, 1]) == sign(grads[3, 1]))
                if sign(grads[1, 1]) == 1
                    g1 = minimum(grads[:, 1])
                else
                    g1 = maximum(grads[:, 1])
                end
            end
            if (sign(grads[1, 2]) == sign(grads[2, 2]) && sign(grads[1, 2]) == sign(grads[3,2]))
                if sign(grads[1, 2]) == 1
                    g2 = minimum(grads[:, 2])
                else
                    g2 = maximum(grads[:, 2])
                end
            end
            grad = SVector(g1, g2)
            
        elseif limiter == 2
            grad = SVector(0.f0, 0.f0)
        end
                
        if any(isnan, grad)
            grad = SVector(0.f0, 0.f0)
        end
        recon_gradient[i, j, 1] = grad[1]
        recon_gradient[i, j, 2] = grad[2]
        
    end
    if i == 9 || i == 3
        #println("Cell $i: \n c: $c, u: $u, u1: $u1, u2: $u2, u3: $u3, c1: $c1, c2: $c2, c3: $c3 \n gradients: $(recon_gradient[i, :, :]) \n")
    end
       
end
    
    
end


function find_neighbours(cell_edge_matrix, edge_cell_matrix, i)

    n1 = 0
    n2 = 0
    n3 = 0
    for j in 1:3
        edge = cell_edge_matrix[i, j]
        if edge_cell_matrix[edge, 1] == i
            if edge_cell_matrix[edge, 2] != 0
                if j == 1
                    n1 = edge_cell_matrix[edge, 2]
                elseif j == 2
                    n2 = edge_cell_matrix[edge, 2]
                elseif j == 3
                    n3 = edge_cell_matrix[edge, 2]
                end
            end
        else
            if j == 1
                n1 = edge_cell_matrix[edge, 1]
            elseif j == 2
                n2 = edge_cell_matrix[edge, 1]
            elseif j == 3
                n3 = edge_cell_matrix[edge, 1]
            end

        end
    end
    return SVector(n1, n2, n3)
end

function check_gradient(gradient, val, val2, c_x, c_y, pt1_x, pt1_y, pt2_x, pt2_y)::Bool
    edge_centre = SVector((pt1_x + pt2_x) / 2, (pt1_y + pt2_y) / 2)
    diff = SVector(edge_centre[1] - c_x, edge_centre[2] - c_y)
    
    edge_val = val + gradient[1] * diff[1] + gradient[2] * diff[2]
    if edge_val >= val && edge_val <= val2
        return true
    elseif edge_val <= val && edge_val >= val2
        return true
    else
        return false
    end
end