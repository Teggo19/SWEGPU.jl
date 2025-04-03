using JLD
# Function to convert regular triangle grid to regular cartesian grid by averaging two and two cells
# and save it to file
function convert_to_cartesian(input, n_x, n_y, filename)
    n_tri = size(input)[1]
    n_cart = div(n_tri, 2)

    res = Array{Float64}(undef, n_x, n_y, 3)

    for i in 1:div(n_tri, 2)
        x_id = (i-1)%n_x + 1
        y_id = div((i-1), n_x) + 1
        res[x_id, y_id, 1] = 0.5*(input[i*2-1, 1] + input[i*2, 1])
        res[x_id, y_id, 2:3] = input[i*2-1, 2:3] + input[i*2, 2:3]
    end

    save("tmp/$filename.jld", "res", res)
    return res
end