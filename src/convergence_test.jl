using GLMakie
using Polynomials

include("Mesh/structured.jl")
include("Mesh/reading.jl")
include("Solver/SWE_solver.jl")
include("Solver/bc.jl")

function convergence_test(baseline, baseline_n, N_list, T, initial_function, title; backend="cpu", bc=neumannBC, time_stepper=0, CFL=0.5f0, altmesh=false, limiter=0, quad_order=3, userecon=true, top_func=0, spaceType=Float32)
    # baseline is the baseline result to compare against
    # N_list is a list of grid sizes to test
    # T is the time to run the simulation for
    # initial_function is a function that takes in a point and returns the initial condition
    # title is the title of the plot
    diffs = zeros(length(N_list))

    for (i, n) in enumerate(N_list)
        if top_func==0
            edges, cells = make_structured_mesh(n, n, spaceType, Int64, alternative=altmesh)
        else
            edges, cells = make_structured_mesh_with_topography(n, n, Float32, Int64, top_func)
        end
        
        #initial = hcat([initial_function(cell.centroid) for cell in cells]...)'
        initial = quadrature(initial_function, cells; order=quad_order)

        res = SWE_solver(cells, edges, T, initial; backend=backend, bc=bc, time_stepper=time_stepper, CFL=CFL, limiter=limiter)
        if userecon
            recon = make_reconstructions(cells, edges, res; limiter=limiter)
            res_refined, _ = refine_structured_grid(res, n, baseline_n, recon)
        else
            res_refined = refine_structured_grid(res, n, baseline_n; altmesh=altmesh)
        end
        
        diffs[i] = L1(res_refined, baseline)
    end

    f = Figure()

    ax1 = Axis(f[1, 1], title=title, xlabel="N", ylabel="L1 error", xscale=log10, yscale=log10)
    
    order_of_convergence = fit(log.(N_list), log.(diffs), 1)
    # Plot the line of best fit
    x = range(minimum(N_list), stop=maximum(N_list), length=100)
    y = exp(order_of_convergence[0]) .* x .^ order_of_convergence[1]
    lines!(ax1, x, y, color=:blue,linestyle=:dash, label="Best fit line: y = $(round(exp(order_of_convergence[0]), digits=2))N^$(round(order_of_convergence[1], digits=2))")
    lines!(ax1, N_list, diffs, color=:red, label="L1 error")
    scatter!(ax1, N_list, diffs, color=:red, markersize=20)
    # Add the legend
    f[1, 2] = Legend(f, ax1, "Plots")
    return N_list, diffs, f, baseline_n, order_of_convergence[1]
end

function convergence_test(N_list, T, initial_function, title; backend="cpu", bc=neumannBC, reconstruction=0)
    baseline_n = N_list[end]

    edges_baseline, cells_baseline = make_structured_mesh(baseline_n, baseline_n, Float32, Int64)

        
    #initial0 = hcat([initial_function(cell.centroid) for cell in cells0]...)'
    initial = quadrature(initial_function, cells_baseline)
    #initial = refine_structured_grid(initial0, N_list[1], baseline_n)

    baseline = SWE_solver(cells_baseline, edges_baseline, T, initial; backend=backend, bc=bc, reconstruction=reconstruction)
    #if reconstruction == 1
    #    recon_baseline = make_reconstructions(cells_baseline, edges_baseline, baseline)
    #end
    

    diffs = zeros(length(N_list)-1)
    for (i, n) in enumerate(N_list[1:end-1])
        edges, cells = make_structured_mesh(n, n, Float32, Int64)
        
        #initial = refine_structured_grid(initial0, N_list[1], n)
        initial = quadrature(initial_function, cells)

        res = SWE_solver(cells, edges, T, initial; backend=backend, bc=bc, reconstruction=reconstruction)
        #=if reconstruction == 1
            recon = make_reconstructions(cells, edges, res)
            res_refined, recon_refined = refine_structured_grid(res, n, baseline_n, recon)
            diffs[i] = L1_quadrature_recon(res_refined, baseline, recon_refined, recon_baseline, cells_baseline)
        else=#
        res_refined = refine_structured_grid(res, n, baseline_n)

        diffs[i] = L1(res_refined, baseline)
        #end

    end

    f = Figure()

    ax1 = Axis(f[1, 1], title=title, xlabel="N", ylabel="L1 error", xscale=log10, yscale=log10)
    lines!(ax1, N_list[1:end-1], diffs, color=:blue, label="L1 error")
    scatter!(ax1, N_list[1:end-1], diffs, color=:blue, markersize=5)
    order_of_convergence = fit(log.(N_list[1:end-1]), log.(diffs), 1)
    # Plot the line of best fit
    x = range(minimum(N_list[1:end-1]), stop=maximum(N_list[1:end-1]), length=100)
    y = exp(order_of_convergence[0]) .* x .^ order_of_convergence[1]
    lines!(ax1, x, y, color=:red, label="Best fit line: y = $(round(exp(order_of_convergence[0]), digits=2))N^$(round(order_of_convergence[1], digits=2))")
    # Add the legend
    f[1, 2] = Legend(f, ax1, "Plots")
    return N_list[1:end-1], diffs, f, baseline_n, order_of_convergence[1]
end


function L1(res1, res2)
    return sum(abs.(res1 .- res2)./size(res1)[1])
end

function L1(res)
    return sum(abs.(res))/size(res)[1]
end

function refine_structured_grid(result, n_old, n_new, reconstruction)
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
    old_recon = reconstruction
    new_res = copy(old_res)
    new_recon = copy(old_recon)
    old_dimension = n_old
    for c in 1:count

        new_dimension = old_dimension*2
        dx = 1/new_dimension
        new_size = new_dimension*new_dimension*2
        new_res = zeros(type, new_size, 3)
        new_recon = zeros(type, new_size, 3, 2)
        for i in 1:old_dimension*old_dimension
            x_coord = (i-1)%old_dimension + 1
            y_coord = div(i-1, old_dimension) 
            if ((x_coord+y_coord)%2) == 1
                if old_recon[2*i-1, :, 1].*(-1/3*dx) + old_recon[2*i-1, :, 2].*(-2/3*dx) != 0
                    #println("Warning: Reconstruction for cell $(2*i-1) is not zero at ($x_coord, $y_coord) point 1. Difference is $(old_recon[2*i-1, :, 1].*(-1/3*dx) + old_recon[2*i-1, :, 2].*(-2/3*dx))")
                end
                #Split 1st cell
                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 1].*(-1/3*dx) + old_recon[2*i-1, :, 2].*(-2/3*dx)
                new_recon[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i-1, :, :]

                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 1].*(-1/3*dx)
                new_recon[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord)+ 2*new_dimension, :, :] = old_recon[2*i-1, :, :]

                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 2].*(1/3*dx)
                new_recon[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord)+ 2*new_dimension, :, :] = old_recon[2*i-1, :, :]

                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 1].*(2/3*dx) + old_recon[2*i-1, :, 2].*(1/3*dx)
                new_recon[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord)+ 2*new_dimension, :, :] = old_recon[2*i-1, :, :]

                #Split 2nd cell
                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord), :]= old_res[2*i, :] + old_recon[2*i, :, 1].*(-2/3*dx) + old_recon[2*i, :, 2].*(-1/3*dx)
                new_recon[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i, :, :]

                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord), :]= old_res[2*i, :] + old_recon[2*i, :, 2].*(-1/3*dx)
                new_recon[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i, :, :]

                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord), :]= old_res[2*i, :] + old_recon[2*i, :, 1].*(1/3*dx)
                new_recon[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i, :, :]

                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :] + old_recon[2*i, :, 1].*(1/3*dx) + old_recon[2*i, :, 2].*(2/3*dx)
                new_recon[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord)+ 2*new_dimension, :, :] = old_recon[2*i, :, :]
            else
                #Split 1st cell
                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 1].*(-1/3*dx)
                new_recon[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i-1, :, :]

                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 2].*(-1/3*dx)
                new_recon[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i-1, :, :]

                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord), :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 1].*(2/3*dx) + old_recon[2*i-1, :, 2].*(-1/3*dx)
                new_recon[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i-1, :, :]

                new_res[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i-1, :] + old_recon[2*i-1, :, 1].*(-1/3*dx) + old_recon[2*i-1, :, 2].*(2/3*dx)
                new_recon[4*(x_coord-1) + 1 + 4*new_dimension*(y_coord)+ 2*new_dimension, :, :] = old_recon[2*i-1, :, :]

                #Split 2nd cell
                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord), :]= old_res[2*i, :] + old_recon[2*i, :, 1].*(1/3*dx) + old_recon[2*i, :, 2].*(-2/3*dx)
                new_recon[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord), :, :] = old_recon[2*i, :, :]

                new_res[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :] + old_recon[2*i, :, 1].*(-2/3*dx) + old_recon[2*i, :, 2].*(1/3*dx)
                new_recon[4*(x_coord-1) + 2 + 4*new_dimension*(y_coord)+ 2*new_dimension, :, :] = old_recon[2*i, :, :]

                new_res[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :] + old_recon[2*i, :, 2].*(1/3*dx)
                new_recon[4*(x_coord-1) + 3 + 4*new_dimension*(y_coord)+ 2*new_dimension, :, :] = old_recon[2*i, :, :]

                new_res[4*(x_coord-1) + 4 + 4*new_dimension*(y_coord)+ 2*new_dimension, :]= old_res[2*i, :] + old_recon[2*i, :, 1].*(1/3*dx)
            end

        end
        old_dimension = new_dimension
        old_res = copy(new_res)
        old_recon = copy(new_recon)
    end
    
    return new_res, new_recon
end

function refine_structured_grid(result, n_old, n_new; altmesh=false)
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
            if ((x_coord+y_coord)%2) == 1 || altmesh
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

# function to compare runtime of CPU and GPU
function compare_runtime(N_list, T, initial_function, title; time_stepper=1, bc=neumannBC)
    cpu_list = zeros(length(N_list))
    gpu_list = zeros(length(N_list))

    for (i, n) in enumerate(N_list)
        edges, cells = make_structured_mesh(n, n, Float32, Int64)
        
        initial = quadrature(initial_function, cells)

        cpu_list[i] = SWE_solver(cells, edges, T, initial; backend="CPU", bc=bc, time_stepper=time_stepper, return_runtime=true)[2]
        gpu_list[i] = SWE_solver(cells, edges, T, initial; backend="CUDA", bc=bc, time_stepper=time_stepper, return_runtime=true)[2]
    end
    f = Figure()
    ax1 = Axis(f[1, 1], title=title, xlabel="N", ylabel="Runtime (s)", xscale=log10, yscale=log10)
    lines!(ax1, N_list, cpu_list, color=:blue, label="CPU runtime")
    scatter!(ax1, N_list, cpu_list, color=:blue, markersize=10)
    lines!(ax1, N_list, gpu_list, color=:red, label="GPU runtime")
    scatter!(ax1, N_list, gpu_list, color=:red, markersize=10)
    # Add the legend
    f[1, 2] = Legend(f, ax1, "Plots")
    return f, cpu_list, gpu_list
end


function quadrature(f, cells; order=3)
    T = eltype(cells[1].centroid)
    res = zeros(T, length(cells), 3)
    #coords = [[4/6, 1/6, 1/6], [1/6, 4/6, 1/6], [1/6, 1/6, 4/6]]
    #weights = [1/3, 1/3, 1/3]
    if order == 1
        coords = [[1/3, 1/3, 1/3]]
        weights = [1.0]
    elseif order == 3
        coords = [[1/3, 1/3, 1/3], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2]]
        weights = [-0.5625, 0.5208333333333333, 0.5208333333333333, 0.5208333333333333]
    end
    for (i, cell) in enumerate(cells)
        pt1 = cell.points[:, 1]
        pt2 = cell.points[:, 2]
        pt3 = cell.points[:, 3]
        
        for (j, coord) in enumerate(coords)
            pt = pt1 .* coord[1] + pt2 .* coord[2] + pt3 .* coord[3]
            res[i, :] += (weights[j] .* f(pt))
        end
        res[i, 1] -= cell.centroid[3]
    end
    res[:, 2:3] = res[:, 2:3] .* res[:, 1]
    return res
end

function quadrature_diff(inp::Vector{T}, f, cells) where {T<:Real}
    res = zeros(T, length(cells), 3)
    coords = [[1/3, 1/3, 1/3], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2]]
    weights = [-0.5625, 0.5208333333333333, 0.5208333333333333, 0.5208333333333333]

    for (i, cell) in enumerate(cells)
        pt1 = cell.pts[:, 1]
        pt2 = cell.pts[:, 2]
        pt3 = cell.pts[:, 3]
        area = cell.area
        for (j, coord) in enumerate(coords)
            pt = pt1 .* coord[1] + pt2 .* coord[2] + pt3 .* coord[3]
            inp_res = inp[cell.ind]

            res[i] += weights[j] * abs(f(pt) - inp_res) * area
        end
    end
    return res
end

include("visualization.jl")
function quadrature_diff_with_recon(inp::Vector{T}, f, cells, recon) where {T<:Real}
    res = zeros(T, length(cells))
    coords = [[1/3, 1/3, 1/3], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2]]
    weights = [-0.5625, 0.5208333333333333, 0.5208333333333333, 0.5208333333333333]
    for (i, cell) in enumerate(cells)
        pt1 = cell.points[:, 1]
        pt2 = cell.points[:, 2]
        pt3 = cell.points[:, 3]
        area = cell.area
        for (j, coord) in enumerate(coords)
            pt = pt1 .* coord[1] + pt2 .* coord[2] + pt3 .* coord[3]
            inp_res = inp[i] + recon[i, 1] * (pt[1] - cell.centroid[1]) + recon[i, 2] * (pt[2] - cell.centroid[2])

            res[i] += weights[j] * abs(f(pt) - inp_res) * area
        end
    end
    return res
end


function L1_quadrature_recon(inp1::Matrix{T}, inp2::Matrix{T}, recon1, recon2, cells) where {T<:Real}
    res = zeros(T, length(cells))
    coords = [[1/3, 1/3, 1/3], [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2]]
    weights = [-0.5625, 0.5208333333333333, 0.5208333333333333, 0.5208333333333333]


    for (i, cell) in enumerate(cells)
        pt1 = cell.points[:, 1]
        pt2 = cell.points[:, 2]
        pt3 = cell.points[:, 3]
        area = cell.area
        for (j, coord) in enumerate(coords)
            pt = pt1 .* coord[1] + pt2 .* coord[2] + pt3 .* coord[3]
            inp_res1 = inp1[i, :] + recon1[i, :, 1] .* (pt[1] - cell.centroid[1]) + recon1[i, :,  2] .* (pt[2] - cell.centroid[2])
            inp_res2 = inp2[i, :] + recon2[i, :, 1] .* (pt[1] - cell.centroid[1]) + recon2[i, :, 2] .* (pt[2] - cell.centroid[2])

            res[i] += weights[j] * sqrt(sum(((inp_res1 - inp_res2).^2))) * area
        end
    end
    return sum(res)
end

function L1_quadrature(inp1::Vector{T}, inp2::Vector{T}, cells, edges) where {T<:Real}
    res = zeros(T, length(cells))
    
    for (i, cell) in enumerate(cells)
        area = cell.area

        res[i] += abs(inp1[i] - inp2[i]) * area
        
    end
    return sum(res)
end
