

function SWE_solver(cells, edges, T, initial)
    # Set initial conditions
    n_edges = length(edges)
    t = 0

    u = initial[:, 0]
    v = initial[:, 1]
    h = initial[:, 2]

    # Loop over time
    while t < T
        # Calculate time step

        # Loop over edges
        for edge in edges
            # Calculate flux
        end

        # Loop over cells
        for cell in cells
            # Calculate new values
        end
    end

    return
end



