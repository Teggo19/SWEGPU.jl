using ProgressMeter

function make_mesh(n_x, n_y)
    #p = Progress(Int64((n_x+1)*(n_y+1)+ n_x*n_y))
    #mesh = Array{Float64, 3}(undef, n_x+1, n_y+1, 3)
    points = Array{Float64, 2}(undef, 3, (n_x+1)*(n_y+1))
    for i in 1:n_x+1
        for j in 1:n_y+1
            #mesh[i, j, 1] = (i-1)/n_x
            #mesh[i, j, 2] = (j-1)/n_y
            #mesh[i, j, 3] = 0.0  # Assuming the third dimension is initially zero

            points[:, i+(j-1)*(n_x+1)] = [(i-1)/n_x, (j-1)/n_y, 0.0]
            #update!(p, n_y*(i-1) + j)
        end
    end

    cells = Array{Int64, 2}(undef, 3, 2*n_x*n_y)

    for j in 1:n_y
        for i in 1:n_x
            global_id = (j-1)*n_x + i
            if (j+i)%2 == 0
                cells[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
                cells[:, 2*global_id] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1)]
            else
                cells[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i + j*(n_x+1)]
                cells[:, 2*global_id] = [i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
            end
        end

        #update!(p, (n_x+1)*(n_y+1)+ i)
    end

    return points, cells
end