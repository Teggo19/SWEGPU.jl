using ProgressMeter
include("reading.jl")
include("save_meshes.jl")

function make_structured_mesh(n_x, n_y, spaceType, indType)
    #p = Progress(Int64((n_x+1)*(n_y+1)+ n_x*n_y))
    #mesh = Array{Float64, 3}(undef, n_x+1, n_y+1, 3)
    filename = "structs/structured_$(n_x)x$(n_y)_$(spaceType)_$(indType)"
    if isfile("tmp/$filename.jld")
        println("Loading mesh from tmp/$filename.jld")
        return load_structures(filename)
    end

    points = Array{spaceType, 2}(undef, 3, (n_x+1)*(n_y+1))
    for i in 1:n_x+1
        for j in 1:n_y+1
            #mesh[i, j, 1] = (i-1)/n_x
            #mesh[i, j, 2] = (j-1)/n_y
            #mesh[i, j, 3] = 0.0  # Assuming the third dimension is initially zero

            points[:, i+(j-1)*(n_x+1)] = [(i-1)/n_x, (j-1)/n_y, 0.0]
            #update!(p, n_y*(i-1) + j)
        end
    end

    faces = Array{indType, 2}(undef, 3, 2*n_x*n_y)

    for j in 1:n_y
        for i in 1:n_x
            global_id = (j-1)*n_x + i
            if (j+i)%2 == 0
                faces[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
                faces[:, 2*global_id] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1)]
            else
                faces[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i + j*(n_x+1)]
                faces[:, 2*global_id] = [i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
            end
        end

        #update!(p, (n_x+1)*(n_y+1)+ i)
    end

    edges, cells = generate_mesh(points, faces, spaceType, indType)
    println("Saving mesh to tmp/$filename.jld")
    save_structures(cells, edges, filename)
    return edges, cells
end

function make_structured_mesh_with_topography(n_x, n_y, spaceType, indType, topography)
    #p = Progress(Int64((n_x+1)*(n_y+1)+ n_x*n_y))
    #mesh = Array{Float64, 3}(undef, n_x+1, n_y+1, 3)
    filename = "structs/structured_$(n_x)x$(n_y)_$(spaceType)_$(indType)"
    if isfile("tmp/$filename.jld")
        println("Loading mesh from tmp/$filename.jld")
        return load_structures_with_topography(filename, topography)
    end

    points = Array{spaceType, 2}(undef, 3, (n_x+1)*(n_y+1))
    for i in 1:n_x+1
        for j in 1:n_y+1
            #mesh[i, j, 1] = (i-1)/n_x
            #mesh[i, j, 2] = (j-1)/n_y
            #mesh[i, j, 3] = 0.0  # Assuming the third dimension is initially zero

            points[:, i+(j-1)*(n_x+1)] = [(i-1)/n_x, (j-1)/n_y, 0.0]
            #update!(p, n_y*(i-1) + j)
        end
    end

    faces = Array{indType, 2}(undef, 3, 2*n_x*n_y)

    for j in 1:n_y
        for i in 1:n_x
            global_id = (j-1)*n_x + i
            if (j+i)%2 == 0
                faces[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
                faces[:, 2*global_id] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1)]
            else
                faces[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i + j*(n_x+1)]
                faces[:, 2*global_id] = [i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
            end
        end

        #update!(p, (n_x+1)*(n_y+1)+ i)
    end

    edges, cells = generate_mesh(points, faces, spaceType, indType)
    println("Saving mesh to tmp/$filename.jld")
    save_structures(cells, edges, filename)

    return load_structures_with_topography(filename, topography)
end


include("../MeshReordering/vertex_cache_alg.jl")
function make_structured_reordered_mesh(n_x, n_y, spaceType, indType)
    #p = Progress(Int64((n_x+1)*(n_y+1)+ n_x*n_y))
    #mesh = Array{Float64, 3}(undef, n_x+1, n_y+1, 3)
    filename = "structs/structured_$(n_x)x$(n_y)_$(spaceType)_$(indType)_reordered"
    if isfile("tmp/$filename.jld")
        println("Loading mesh from tmp/$filename.jld")
        return load_structures(filename)
    end

    points = Array{spaceType, 2}(undef, 3, (n_x+1)*(n_y+1))
    for i in 1:n_x+1
        for j in 1:n_y+1
            #mesh[i, j, 1] = (i-1)/n_x
            #mesh[i, j, 2] = (j-1)/n_y
            #mesh[i, j, 3] = 0.0  # Assuming the third dimension is initially zero

            points[:, i+(j-1)*(n_x+1)] = [(i-1)/n_x, (j-1)/n_y, 0.0]
            #update!(p, n_y*(i-1) + j)
        end
    end

    faces = Array{indType, 2}(undef, 3, 2*n_x*n_y)

    for j in 1:n_y
        for i in 1:n_x
            global_id = (j-1)*n_x + i
            if (j+i)%2 == 0
                faces[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
                faces[:, 2*global_id] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1)]
            else
                faces[:, 2*global_id-1] = [i + (j-1)*(n_x+1), i+1 + (j-1)*(n_x+1), i + j*(n_x+1)]
                faces[:, 2*global_id] = [i+1 + (j-1)*(n_x+1), i+1 + j*(n_x+1), i + j*(n_x+1)]
            end
        end

        #update!(p, (n_x+1)*(n_y+1)+ i)
    end
    #return points, faces
    new_points, new_faces = reorder_triangular_grid(points, faces)
    edges, cells = generate_mesh(new_points, new_faces, spaceType, indType)
    #println("Saving mesh to tmp/$filename.jld")
    #save_structures(cells, edges, filename)
    return edges, cells
end