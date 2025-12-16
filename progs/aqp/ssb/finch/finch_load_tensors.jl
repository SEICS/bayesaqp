# using HDF5
using Finch
using TensorMarket
# using NPZ # Can't load python objects. Needs manual conversion
using PyCall

# Import our testing module (handle both relative and absolute paths)
script_dir = dirname(@__FILE__)
include(joinpath(script_dir, "conversion_test.jl"))
using .ConversionTest
@pyimport pickle
@pyimport gzip
np = pyimport("numpy")

# cpds = Finch.bspread(h5open("progs/aqp/ssb/finch/cpds.bsp.h5","r")) #official package not up-to-date, author planned to fix
# cpds = Finch.fread("progs/aqp/ssb/finch/test.mtx") # works

"""
    test_symmetric_conversion()

Test the symmetric to general matrix conversion function for correctness.
"""
function test_symmetric_conversion()
    println("Testing symmetric matrix conversion...")
    result = ConversionTest.test_conversion_correctness(convert_symmetric_to_general, verbose=true)
    return result
end

function convert_symmetric_to_general(input_file::String, output_file::String)
    """Convert a symmetric MTX file to general format"""
    lines = readlines(input_file)
    
    # Find the header line and modify it
    header_idx = 1
    while header_idx <= length(lines) && !startswith(lines[header_idx], "%%MatrixMarket")
        header_idx += 1
    end
    
    if header_idx > length(lines)
        error("No MatrixMarket header found in $input_file")
    end
    
    # Replace "symmetric" with "general" in header
    header = replace(lines[header_idx], "symmetric" => "general")
    
    # Find the size line first
    size_line_idx = header_idx + 1
    while size_line_idx <= length(lines) && (startswith(lines[size_line_idx], "%") || !occursin(r"^\s*\d+\s+\d+\s+\d+\s*$", lines[size_line_idx]))
        size_line_idx += 1
    end
    
    # Write the converted file
    open(output_file, "w") do f
        # Write modified header
        println(f, header)
        
        # Write comments and size line
        for i in (header_idx+1):size_line_idx
            println(f, lines[i])
        end
        
        # For symmetric matrices, we need to expand the entries
        # Read the data entries and expand symmetric entries
        entries = []
        for i in (size_line_idx+1):length(lines)
            line = strip(lines[i])
            if !isempty(line)
                parts = split(line)
                if length(parts) >= 3
                    row = parse(Int, parts[1])
                    col = parse(Int, parts[2])
                    val = parts[3]
                    
                    # Add the original entry
                    push!(entries, (row, col, val))
                    
                    # Add the symmetric entry if it's not on the diagonal
                    if row != col
                        push!(entries, (col, row, val))
                    end
                end
            end
        end
        
        # Update the size line with correct number of entries
        size_parts = split(lines[size_line_idx])
        if length(size_parts) >= 3
            # Update the third number (number of entries)
            size_parts[3] = string(length(entries))
            # Rewrite the size line
            seek(f, 0)
            println(f, header)
            for i in (header_idx+1):(size_line_idx-1)
                println(f, lines[i])
            end
            println(f, join(size_parts, " "))
        end
        
        # Write all entries
        for (row, col, val) in entries
            println(f, "$row $col $val")
        end
    end
end

function finch_load_sparse(folder_path) # need folder path where all sparse matrices in .mtx format are stored
    println("Searching for sparse matrices .mtx files from: $folder_path")
    
    # Check if folder exists
    if !isdir(folder_path)
        error("Directory '$folder_path' does not exist")
    end
    
    found_files = readdir(folder_path)
    mtx_files = filter(f -> endswith(f, ".mtx"), found_files)
    
    println("Found $(length(mtx_files)) .mtx files out of $(length(found_files)) total files")
    
    # Read each .mtx file as a matrix in Finch.Tensor format
    finch_tensors = Dict{String, Any}()
    mtx_count = 0
    
    for (i, ff) in enumerate(mtx_files)
        try
            println("[$i/$(length(mtx_files))] Loading: $ff")
            tensor_name = replace(ff, ".mtx" => "") # .mtx is in format of tensor_name.mtx, remove .mtx to reference tensor_name
            file_path = "$folder_path/$ff"
            
            # Try to load directly first
            try
                finch_tensors[tensor_name] = Finch.fread(file_path)
                mtx_count += 1
                println("  ✓ Successfully loaded as Finch.Tensor")
            catch e
                # Check if it's a symmetric matrix error
                if occursin("symmetric", string(e))
                    println("  → Detected symmetric matrix, converting to general format...") # TensorMarket.jl doesn't support symmetric matrices
                    temp_file = tempname() * ".mtx" # Create temporary converted file
                    convert_symmetric_to_general(file_path, temp_file)
                    
                    try
                        finch_tensors[tensor_name] = Finch.fread(temp_file)
                        mtx_count += 1
                        println("  ✓ Successfully converted and loaded as Finch.Tensor")
                        
                        rm(temp_file) # Clean up temp file
                    catch conv_error
                        println("  ✗ Error loading converted file: $conv_error")
                        isfile(temp_file) && rm(temp_file) # Clean up temp file
                        rethrow(conv_error)
                    end
                else
                    rethrow(e)
                end
            end
        catch e
            println("  ✗ Error loading $ff: $e")
        end
    end
    
    println("\n" * "="^50)
    println("Summary:")
    println("Successfully converted $mtx_count/$(length(mtx_files)) .mtx files to Finch.Tensor format")
    
    if mtx_count > 0
        ex_name = first(keys(finch_tensors))
        ex_tensor = finch_tensors[ex_name]
        println("\nExample tensor '$ex_name':")
        println("  Type: $(typeof(ex_tensor))")
        println("  Size: $(size(ex_tensor))")
    end
    
    return finch_tensors
end

function finch_load_dense(npz_path) # .npy or .npz path
    dense = np.load("$npz_path", allow_pickle=true)
    expected = length(dense.files)
    
    # Convert each numpy array to Finch.Tensor
    finch_tensors = Dict{String, Any}()
    
    # Process ALL tensors in the NPZ file
    processed_count = 0
    none_count = 0
    for tensor_name in dense.files
        processed_count += 1
        
        # Get the numpy array using get_item method
        numpy_array = dense.__getitem__(tensor_name)
        
        println("\n[$processed_count/170] Processing: $tensor_name")
        println("  Shape: $(size(numpy_array))")
        println("  Type: $(typeof(numpy_array))")
        
        try
            # Handle different types of Python objects
            if typeof(numpy_array) <: Array{PyObject}
                # This is a Python object array - need to convert
                println("  → Converting Python object array...")
                
                # Try to extract the actual Python object and convert it
                if ndims(numpy_array) == 0
                    # Scalar PyObject - extract the value
                    py_obj = numpy_array[]
                    println("  → Python object type: $(py_obj.__class__)")
                    
                    # Try to convert to Julia array
                    temp_arr = try
                        # Check if it's None
                        if py_obj == pybuiltin("None")
                            # Handle None/null values - create empty array or skip
                            none_count += 1
                            println("  → Skipping None/null value")
                            nothing
                        # If it's a numpy array, convert it
                        elseif hasproperty(py_obj, :shape)
                            convert(Array, py_obj)
                        # If it's a dictionary, extract keys or values
                        elseif hasproperty(py_obj, :keys)
                            # Convert dict to array of keys and values
                            keys_list = collect(py_obj.keys())
                            values_list = collect(py_obj.values())
                            println("  → Dictionary with $(length(keys_list)) keys")
                            
                            # Debug: show first few key-value pairs
                            println("  → Sample key-value pairs:")
                            for i in 1:min(5, length(keys_list))
                                println("    $(keys_list[i]) => $(values_list[i])")
                            end
                            
                            # Convert to proper Julia arrays
                            try
                                # Extract actual values from the lists/arrays
                                extracted_values = []
                                for v in values_list
                                    if isa(v, Array) || isa(v, Vector) || hasproperty(v, :__len__)
                                        # If it's a list/array, take the first element
                                        if length(v) > 0
                                            push!(extracted_values, v[1])
                                        else
                                            push!(extracted_values, 0)  # Default for empty
                                        end
                                    else
                                        # If it's already a scalar, use it directly
                                        push!(extracted_values, v)
                                    end
                                end
                                
                                # Now try to convert to integers
                                int_values = [convert(Int, v) for v in extracted_values]
                                
                                println("  → Successfully converted to integer array of actual values")
                                int_values
                            catch int_error
                                println("  → Integer conversion failed: $int_error")
                                try
                                    # Try to convert to floats
                                    extracted_values = []
                                    for v in values_list
                                        if isa(v, Array) || isa(v, Vector) || hasproperty(v, :__len__)
                                            if length(v) > 0
                                                push!(extracted_values, v[1])
                                            else
                                                push!(extracted_values, 0.0)
                                            end
                                        else
                                            push!(extracted_values, v)
                                        end
                                    end
                                    
                                    float_values = [convert(Float64, v) for v in extracted_values]
                                    println("  → Successfully converted to float array of actual values")
                                    float_values
                                catch float_error
                                    println("  → Float conversion failed: $float_error")
                                    # Create a simple index array instead
                                    println("  → Creating index array 1:$(length(keys_list))")
                                    collect(1:length(keys_list))
                                end
                            end
                        # If it's a list, convert to array
                        elseif hasproperty(py_obj, :__len__)
                            collect(py_obj)
                        else
                            # Try generic conversion
                            println("  → Attempting generic conversion...")
                            convert(Array, py_obj)
                        end
                    catch conv_error
                        println("  → Could not convert Python object: $conv_error")
                        # Try to get string representation
                        try
                            [string(py_obj)]
                        catch
                            nothing
                        end
                    end
                    
                    if temp_arr !== nothing
                        finch_tensor = Tensor(temp_arr)
                        finch_tensors[tensor_name] = finch_tensor
                        println("  ✓ Successfully converted Python object to Finch.Tensor")
                        println("  ✓ Converted shape: $(size(temp_arr))")
                        println("  ✓ Finch tensor type: $(typeof(finch_tensor))")
                    else
                        println("  ✗ Could not convert Python object")
                    end
                else
                    # Multi-dimensional PyObject array
                    println("  → Multi-dimensional Python object array")
                    # Try to convert each element
                    converted_elements = []
                    for py_obj in numpy_array
                        try
                            if hasproperty(py_obj, :shape)
                                push!(converted_elements, convert(Array, py_obj))
                            else
                                push!(converted_elements, py_obj)
                            end
                        catch
                            push!(converted_elements, string(py_obj))
                        end
                    end
                    
                    if !isempty(converted_elements)
                        temp_arr = reshape(converted_elements, size(numpy_array))
                        finch_tensor = Tensor(temp_arr)
                        finch_tensors[tensor_name] = finch_tensor
                        println("  ✓ Successfully converted multi-dim Python object to Finch.Tensor")
                        println("  ✓ Finch tensor type: $(typeof(finch_tensor))")
                    else
                        println("  ✗ Could not convert multi-dimensional Python object")
                    end
                end
            else
                # Regular Julia array - convert directly
                finch_tensor = Tensor(numpy_array)
                finch_tensors[tensor_name] = finch_tensor
                println("  ✓ Successfully converted to Finch.Tensor")
                println("  ✓ Finch tensor type: $(typeof(finch_tensor))")
            end
            
        catch e
            println("  ✗ Error converting $tensor_name: $e")
        end
    end
    
    println("\n" * "="^50)
    println("Summary:")
    println("In total, processed $(length(finch_tensors)+none_count)/$expected tensors.")
    println("Successfully converted $(length(finch_tensors))/$expected tensors to Finch format.")
    println("$none_count/$expected None/null values are skipped.")
    
    # Show the first tensor as an example
    if !isempty(finch_tensors)
        ex_name = first(keys(finch_tensors))
        ex_tensor = finch_tensors[ex_name]
        
        println("\nExample tensor '$ex_name':")
        println("  Type: $(typeof(ex_tensor))")
        println("  Size: $(size(ex_tensor))")
        
        # You can now use this tensor with Finch operations
        # Example: println("  First few elements: ", ex_tensor[1:min(5, length(ex_tensor))])
    end
    return finch_tensors
end

function finch_load_all(folder_path)
    dense = finch_load_dense(joinpath(folder_path, "dense.npz"))
    sparse = finch_load_sparse(folder_path)
    return merge(dense, sparse) # merge will overwrite elements in the front dict with that in the last dict if having same keys
end

# function main()
#     cpds_folder = "progs/aqp/ssb/finch/data"
#     # Load cpds tensors in Finch.Tensor format
#     cpds = finch_load_all(cpds_folder)
#     println(cpds)
# end

# main()