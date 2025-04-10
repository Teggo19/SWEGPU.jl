include("Mesh/structured.jl")
include("Mesh/reading.jl")

function convergence_test(N_list, T, initial_function)
    for n in N_list
        pts, fs = make_structured_mesh(n, n)
        edges, cells = generate_mesh(pts, fs)


    end

end


function merge_cells(result, n_old, n_new)
    #=
    Takes in the result of a structured triangular grid with size n_old x n_old and 
    changes it into a result with a structured n_new x n_new grid
    
    n_new/n_old must be a power of 2
    =#
    type = eltype(result)

    a = div(n_old, n_new)
    count = 0
    while a%2 == 0
        println("a = $a")
        a = div(a, 2)
        count = count + 1
    end
    if a != 1
        print("A $n_old x $n_old mesh cannot be changed to a $n_new x $n_new mesh.")
    end

    old_res = copy(result)
    new_res = copy(old_res)
    old_size = n_old*n_old*2
    for c in 1:count
        # half the mesh size
        new_size = div(n_old*n_old*2, 2)
        new_res = Matrix{type}(undef, new_size, 3)

        for cell in 1:new_size
            new_res[cell, 1] = 0.5*(old_res[2*cell-1, 1] + old_res[2*cell, 1])
            new_res[cell, 2:3] = old_res[2*cell-1, 2:3] + old_res[2*cell, 2:3]
        end

        old_res = copy(new_res)
        old_size = new_size
    end
    
    return new_res
end