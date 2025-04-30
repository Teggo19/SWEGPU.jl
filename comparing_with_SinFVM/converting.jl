using StaticArrays
function cartesian_to_triangular(state)
    # Converts n_x times n_y matrix from cartesian grid to a result for a structured 
    # triangular mesh
    n_x, n_y = size(state)[1:2]
    
    res = Array{Float64}(undef, 2*n_x*n_y, 3)

    for j in 1:n_y
        for i in 1:n_x
            global_index = i + (j-1)*n_x
            res[global_index*2-1, :] = state[i, j]
            res[global_index*2, :] = state[i, j]
        end
    end
    return res
end

# Function to convert regular triangle grid to regular cartesian grid by averaging two and two cells
# and save it to file
function triangular_to_cartesian(input, n_x, n_y)
    n_tri = size(input)[1]

    res = Array{Float64}(undef, n_x, n_y, 3)

    for i in 1:div(n_tri, 2)
        x_id = (i-1)%n_x + 1
        y_id = div((i-1), n_x) + 1
        res[x_id, y_id, 1] = 0.5*(input[i*2-1, 1] + input[i*2, 1])
        res[x_id, y_id, 2:3] = input[i*2-1, 2:3] + input[i*2, 2:3]
    end
    return res
end