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

    sl_t = Slider(fig[2, 1], range = 0:1:n_timesteps, startvalue = 0)
    label = Label(fig[2, 2], text = "t = 0")

    

    viz_array = Array{Any}(undef, n_timesteps)

    t = 0
    U = deepcopy(U0)
    viz_array[1] = visualize_height(U, cells, edges)
    while t < T
        new_t = min(t + dt, T)
        U = SWE_solver(cells, edges, new_t - t, U; bc=bc, backend=backend)
        t = new_t
        viz_array[ceil(Int, t/dt)+1] = visualize_height(U, cells, edges)
    end

    # Create a function to update the visualization
    on(sl_t.value) do t
        show_visuals(t)
    end

    function show_visuals(t)
        label.text = "t = $(t)"

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
        @async for i in 1:n_timesteps
            sleep(dt/simulation_speed)
            show_visuals(i)
            set_close_to!(sl_t, i)
        end
        
    end

    fig
end