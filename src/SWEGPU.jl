module SWEGPU


    include("read_obj.jl")
    include("structs.jl")
    include("Mesh/reading.jl")
    include("Mesh/structured.jl")
    include("visualization.jl")
    include("Solver/SWE_solver.jl")
    include("convergence_test.jl")
    include("Solver/bc.jl")
    include("dynamic_visualization.jl")
end