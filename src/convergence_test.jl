using GLMakie
using Polynomials

include("Mesh/structured.jl")
include("Mesh/reading.jl")
include("Solver/SWE_solver.jl")
include("Solver/bc.jl")

function convergence_test(N_list, T, initial_function, title; backend="cpu", bc=neumannBC, reconstruction=0)
    baseline_n = N_list[end]

    edges, cells = make_structured_mesh(baseline_n, baseline_n, Float32, Int64)
        
    initial = hcat([initial_function(cell.centroid) for cell in cells]...)'

    baseline = SWE_solver(cells, edges, T, initial; backend=backend, bc=bc, reconstruction=reconstruction)

    diffs = zeros(length(N_list)-1)
    for (i, n) in enumerate(N_list[1:end-1])
        edges, cells = make_structured_mesh(n, n, Float32, Int64)
        
        initial = hcat([initial_function(cell.centroid) for cell in cells]...)'

        res = SWE_solver(cells, edges, T, initial; backend=backend, bc=bc, reconstruction=reconstruction)

        res_refined = refine_structured_grid(res, n, baseline_n)

        diffs[i] = L1(res_refined, baseline)

    end

    f = Figure()

    ax1 = Axis(f[1, 1], title=title, xlabel="N", ylabel="L1 error", xscale=log10, yscale=log10)
    lines!(ax1, N_list[1:end-1], diffs, color=:blue, label="L1 error")
    scatter!(ax1, N_list[1:end-1], diffs, color=:blue, markersize=5)
    order_of_convergence = fit(log.(N_list[1:end-1]), log.(diffs), 1)
    # Plot the line of best fit
    x = range(minimum(N_list[1:end-1]), stop=maximum(N_list[1:end-1]), length=100)
    y = exp(order_of_convergence[0]) .* x .^ order_of_convergence[1]
    lines!(ax1, x, y, color=:red, label="Best fit line: y = $(round(order_of_convergence[0], digits=2))N^$(round(order_of_convergence[1], digits=2))")
    # Add the legend
    f[1, 2] = Legend(f, ax1, "Plots")
    return N_list[1:end-1], diffs, f, baseline_n, order_of_convergence[1]
end


function L1(res1, res2)
    return sum(abs.(res1 .- res2))/size(res1)[1]
end

function refine_structured_grid(result, n_old, n_new)
    #=
    Takes in the result of a structured triangular grid with size n_old x n_old and 
    changes it into a result with a structured n_new x n_new grid
    
    n_new/n_old must be a power of 2
    =#
    type = eltype(result)

    a = div(n_new, n_old)
    count = 0
    while a%2 == 0
        a = div(a, 2)
        count = count + 1
    end
    if a != 1
        println("A $n_old x $n_old mesh cannot be changed to a $n_new x $n_new mesh.")
        return
    end

    #= if (n_old%2) == 1
        println("Have not yet implemented support for meshes that are of odd size.")
        return
    end =#
    old_res = copy(result)
    new_res = copy(old_res)
    old_dimension = n_old
    for c in 1:count

        new_dimension = old_dimension*2
        new_size = new_dimension*new_dimension*2
        new_res = zeros(type, new_size, 3)
        for i in 1:old_dimension*old_dimension
            x_coord = (i-1)%old_dimension + 1
            y_coord = div(i-1, old_dimension) 
            if ((x_coord+y_coord)%2) == 1
                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :]
                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :]
                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :]
                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :]
                
                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord), :]= old_res[2*i, :]
                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord), :]= old_res[2*i, :]
                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord), :]= old_res[2*i, :]
                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :]
            else
                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :]
                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :]
                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :]
                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :]
                
                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord), :]= old_res[2*i, :]
                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :]
                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :]
                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :]
            end

        end
        old_dimension = new_dimension
        old_res = copy(new_res)
    end
    
    return new_res
end