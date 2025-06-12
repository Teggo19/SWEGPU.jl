mutable struct Vertex
    id::Int
    tot_triangles::Int
    triangles_not_drawn::Int
    cache_position::Int
    triangles::Vector{Int} # Old triangle IDs
    score::Float64
end

mutable struct Face
    old_id::Int
    drawn::Bool
    score::Float64
    vertices::Vector{Int}
    new_id::Int
end

cache_size = 32

function update_cache!(triangle::Face, vertex_cache::Vector{Int}, vertices::Vector{Vertex}, triangles::Vector{Face})
    for vertex_id in triangle.vertices
        vertex = vertices[vertex_id]
        if vertex.cache_position == -1
            shift_cache!(vertex_cache, cache_size + 3)
        else
            shift_cache!(vertex_cache, vertex.cache_position)
        end
        vertex_cache[1] = vertex_id
        vertex.cache_position = 1
    end

    for (i, vertex_id) in enumerate(vertex_cache)
        if vertex_id == -1
            continue
        end
        vertex = vertices[vertex_id]
        if i > cache_size
            vertex.cache_position = -1
        else
            vertex.cache_position = i
        end
        find_vertex_score!(vertex, triangles)
        
    end

end

function shift_cache!(vertex_cache, index)
    # shift the cache one position to the right until the index
    
    for i in index:-1:2
        vertex_cache[i] = vertex_cache[i-1]
    end
end

function find_vertex_score!(vertex::Vertex, triangles)
    recent_vertex_score = 0.75
    decay_power = 1.5
    valence_boost_scalar = 2.0
    valence_boost_power = -0.5

    old_score = vertex.score

    if vertex.triangles_not_drawn == 0
        return -1.0
    end
    score = 0.0
    if vertex.cache_position == -1
        score = 0.0
    elseif vertex.cache_position <= 3
        score = recent_vertex_score
    else
        scaler = 1.0 / (cache_size - 3)
        score = 1.0 - scaler * (vertex.cache_position - 3)
        score = score^decay_power
    end

    valence_boost = vertex.triangles_not_drawn^valence_boost_power
    score += valence_boost_scalar * valence_boost

    vertex.score = score
    for triangle_id in vertex.triangles
        triangle = triangles[triangle_id]
        triangle.score += vertex.score - old_score        
    end
end

function find_next_triangle!(triangles)
    max_score = -1.0
    best_triangle = nothing
    for triangle in triangles
        if triangle.drawn
            continue
        end
        score = triangle.score
        if score > max_score
            max_score = score
            best_triangle = triangle
        end
    end
    return best_triangle
end

function draw_triangle!(triangle, vertices, new_id)
    triangle.drawn = true
    for vertex_id in triangle.vertices
        vertex = vertices[vertex_id]
        vertex.triangles_not_drawn -= 1
    end
    triangle.new_id = new_id
end

function reorder_triangles!(vertices::Vector{Vertex}, triangles::Vector{Face})
    n_triangles = length(triangles)
    triangles_drawn = 0

    vertex_cache = ones(Int64, cache_size + 3).* -1

    while triangles_drawn < n_triangles
        triangle = find_next_triangle!(triangles)
        #println("Drawing triangle $(triangle.old_id) with new id $(triangles_drawn + 1) and score $(triangle.score)")
        draw_triangle!(triangle, vertices, triangles_drawn + 1)
        
        update_cache!(triangle, vertex_cache, vertices, triangles)
        triangles_drawn += 1
    end
    return triangles
end


function make_vertices_and_triangles(points, faces)
    n_verts = size(points, 2)
    n_tris = size(faces, 2)

    vertices = Vector{Vertex}(undef, n_verts)
    triangles = Vector{Face}(undef, n_tris)

    for i in 1:n_verts
        vertices[i] = Vertex(i, 0, 0, -1, Int[], 0.0)
    end

    for i in 1:n_tris
        triangles[i] = Face(i, false, 0.0, Int[], -1)
        for j in 1:3
            vertex_id = faces[j, i]
            push!(triangles[i].vertices, vertex_id)
            push!(vertices[vertex_id].triangles, i)
            vertices[vertex_id].tot_triangles += 1
            vertices[vertex_id].triangles_not_drawn += 1
        end
    end
    for vert in 1:n_verts
        find_vertex_score!(vertices[vert], triangles)
    end

    return vertices, triangles
end


function reorder_triangular_grid(points, faces)
    vertices, triangles = make_vertices_and_triangles(points, faces)

    reorder_triangles!(vertices, triangles)

    new_faces = Array{Int64, 2}(undef, 3, length(triangles))
    for triangle in triangles
        new_faces[1, triangle.new_id] = triangle.vertices[1]
        new_faces[2, triangle.new_id] = triangle.vertices[2]
        new_faces[3, triangle.new_id] = triangle.vertices[3]
    end
    return points, new_faces
end
