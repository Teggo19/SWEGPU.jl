module SWEGPU


    include("read_obj.jl")
    include("structs.jl")
    include("struct_helper.jl")

    include("Mesh/reading.jl")
    include("Mesh/structured.jl")
    include("Mesh/save_meshes.jl")

    include("visualization.jl")
    include("dynamic_visualization.jl")

    include("Solver/SWE_solver.jl")
    include("convergence_test.jl")
    include("Solver/bc.jl")
    
end