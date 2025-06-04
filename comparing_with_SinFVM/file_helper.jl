using JLD
function read_from_file(filename; input="res")
    result = load("tmp/SinFVM/$filename.jld", input)
    
    return result
end

function write_to_file(array, filename, array_name)
    save("tmp/$filename.jld", array_name, array)
    return 
end