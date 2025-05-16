using GLMakie
using Meshes

include("Solver/bc.jl")
include("visualization.jl")
include("Solver/SWE_solver.jl")


function dynamic_visualization(edges, cells, U0, T, dt ; bc=neumannBC, backend="cpu")
    fig = Figure()
    ax1 = Axis3(fig[1, 1], title="h")
    limits!(ax1, 0, 1, 0, 1, 0, 2)

    n_timesteps = ceil(Int, T/dt)+1
    n_cells = length(cells)

    sl_t = Slider(fig[2, 1], range = 0:1:n_timesteps-1, startvalue = 0)
    label = Label(fig[2, 2], text = "t = 0.00")

    

    viz_array = Array{Any}(undef, n_timesteps)

    t = 0
    U = deepcopy(U0)
    viz_array[1] = visualize_height(U, cells, edges)
    index = 1
    while t < T
        index += 1
        new_t = min(t + dt, T)
        U = SWE_solver(cells, edges, new_t - t, U; bc=bc, backend=backend)
        t = new_t
        viz_array[index] = visualize_height(U, cells, edges)
    end

    # Create a function to update the visualization
    on(sl_t.value) do t
        show_visuals(t)
    end

    function show_visuals(t)
        label.text = "t = "*string(round(t*dt; digits=2))

        #for plot in ax1.scene.plots
        #    delete!(ax1.scene, plot)
        #end
        delete!(ax1.scene, ax1.scene.plots[end])

        viz!(ax1, viz_array[t+1])
    end

    button = Button(fig, label="Play")
    simulation_speed = T/3
    on(button.clicks) do n
        sl_t.value = 0
        @async for i in 1:n_timesteps-1
            sleep(dt/simulation_speed)
            show_visuals(i)
            set_close_to!(sl_t, i)
        end
        
    end

    fig
end

function dynamic_visualization_heatmap(edges, cells, U0, T, dt ; bc=neumannBC, backend="cpu")
    fig = Figure()
    ax1 = Axis(fig[1, 1], title="h")
    limits!(ax1, 0, 1, 0, 1)

    n_timesteps = ceil(Int, T/dt)+1
    n_cells = length(cells)

    sl_t = Slider(fig[2, 1], range = 0:1:n_timesteps-1, startvalue = 0)
    label = Label(fig[2, 2], text = "t = 0.00")

    min_max = [0, maximum(U0[:, 1])]

    viz_array = Array{Any}(undef, n_timesteps)
    color_array = Array{Any}(undef, n_timesteps)

    t = 0
    U = deepcopy(U0)
    viz_array[1], color_array[1] = visualize_heatmap(U, cells)
    viz!(ax1, viz_array[1], color = color_array[1])
    Colorbar(fig[1, 2], limits=(min_max[1], min_max[2]), colormap=:viridis, width=20)
    index=1
    while t < T
        index += 1
        new_t = min(t + dt, T)
        U = SWE_solver(cells, edges, new_t - t, U; bc=bc, backend=backend)
        t = new_t
        viz_array[index], color_array[index] = visualize_heatmap(U, cells)
    end

    # Create a function to update the visualization
    on(sl_t.value) do t
        show_visuals(t)
    end

    function show_visuals(t)
        label.text = "t = "*string(round(t*dt; digits=2))

        #for plot in ax1.scene.plots
        #    delete!(ax1.scene, plot)
        #end
        delete!(ax1.scene, ax1.scene.plots[end])

        viz!(ax1, viz_array[t+1], color=color_array[t+1])
    end

    button = Button(fig, label="Play")
    simulation_speed = T/3
    on(button.clicks) do n
        sl_t.value = 0
        @async for i in 1:n_timesteps-1
            sleep(dt/simulation_speed)
            show_visuals(i)
            set_close_to!(sl_t, i)
        end
        
    end

    fig
end