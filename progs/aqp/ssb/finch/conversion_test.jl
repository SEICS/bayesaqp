"""
Module for testing the correctness of symmetric to general matrix conversion.

This module provides functions to verify that the conversion preserves the 
mathematical properties of symmetric matrices.
"""
module ConversionTest

using SparseArrays
using LinearAlgebra

export test_conversion_correctness, create_test_matrix, verify_symmetry, compare_matrices

"""
    create_test_matrix(filename::String, matrix_type::Symbol=:mixed)

Create a test symmetric matrix file for testing conversion.

# Arguments
- `filename`: Output filename for the test matrix
- `matrix_type`: Type of test matrix to create
  - `:diagonal`: Only diagonal entries
  - `:mixed`: Mix of diagonal and off-diagonal entries
  - `:full`: Full symmetric matrix with many off-diagonal entries

# Returns
- Tuple of (expected_full_matrix, original_entries_count)
"""
function create_test_matrix(filename::String, matrix_type::Symbol=:mixed)
    if matrix_type == :diagonal
        # Create a diagonal matrix (like your actual data)
        open(filename, "w") do f
            println(f, "%%MatrixMarket matrix coordinate real symmetric")
            println(f, "%")
            println(f, "3 3 3")
            println(f, "1 1 5.0")
            println(f, "2 2 10.0") 
            println(f, "3 3 15.0")
        end
        expected_matrix = [5.0 0.0 0.0; 0.0 10.0 0.0; 0.0 0.0 15.0]
        return expected_matrix, 3
        
    elseif matrix_type == :mixed
        # Mixed diagonal and off-diagonal
        open(filename, "w") do f
            println(f, "%%MatrixMarket matrix coordinate real symmetric")
            println(f, "%")
            println(f, "4 4 6")
            println(f, "1 1 1.0")  # diagonal
            println(f, "1 2 2.0")  # off-diagonal
            println(f, "1 3 3.0")  # off-diagonal
            println(f, "2 2 4.0")  # diagonal
            println(f, "2 4 5.0")  # off-diagonal
            println(f, "3 3 6.0")  # diagonal
        end
        expected_matrix = [1.0 2.0 3.0 0.0; 2.0 4.0 0.0 5.0; 3.0 0.0 6.0 0.0; 0.0 5.0 0.0 0.0]
        return expected_matrix, 6
        
    elseif matrix_type == :full
        # Full symmetric matrix
        open(filename, "w") do f
            println(f, "%%MatrixMarket matrix coordinate integer symmetric")
            println(f, "%")
            println(f, "3 3 6")
            println(f, "1 1 1")    # diagonal
            println(f, "1 2 2")    # off-diagonal
            println(f, "1 3 3")    # off-diagonal
            println(f, "2 2 4")    # diagonal
            println(f, "2 3 5")    # off-diagonal
            println(f, "3 3 6")    # diagonal
        end
        expected_matrix = [1 2 3; 2 4 5; 3 5 6]
        return expected_matrix, 6
    else
        error("Unknown matrix_type: $matrix_type. Use :diagonal, :mixed, or :full")
    end
end

"""
    verify_symmetry(matrix::Matrix)

Verify that a matrix is symmetric.

# Arguments
- `matrix`: The matrix to check

# Returns
- `true` if symmetric, `false` otherwise
"""
function verify_symmetry(matrix::Matrix)
    return matrix ≈ matrix'
end

"""
    parse_mtx_to_matrix(filename::String)

Parse an MTX file and convert it to a full Julia matrix.

# Arguments
- `filename`: Path to the MTX file

# Returns
- Full matrix representation
"""
function parse_mtx_to_matrix(filename::String)
    lines = readlines(filename)
    
    # Find header
    header_idx = 1
    while header_idx <= length(lines) && !startswith(lines[header_idx], "%%MatrixMarket")
        header_idx += 1
    end
    
    # Find size line
    size_idx = header_idx + 1
    while size_idx <= length(lines) && (startswith(lines[size_idx], "%") || isempty(strip(lines[size_idx])))
        size_idx += 1
    end
    
    # Parse dimensions
    size_parts = split(lines[size_idx])
    rows, cols, nnz = parse(Int, size_parts[1]), parse(Int, size_parts[2]), parse(Int, size_parts[3])
    
    # Create sparse matrix
    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float64[]
    
    for i in (size_idx+1):length(lines)
        line = strip(lines[i])
        if !isempty(line)
            parts = split(line)
            if length(parts) >= 3
                push!(I_vals, parse(Int, parts[1]))
                push!(J_vals, parse(Int, parts[2]))
                push!(V_vals, parse(Float64, parts[3]))
            end
        end
    end
    
    # Create and convert to full matrix
    sparse_mat = sparse(I_vals, J_vals, V_vals, rows, cols)
    return Matrix(sparse_mat)
end

"""
    compare_matrices(expected::Matrix, actual::Matrix; tolerance::Float64=1e-10)

Compare two matrices for equality within tolerance.

# Arguments
- `expected`: Expected matrix
- `actual`: Actual matrix from conversion
- `tolerance`: Numerical tolerance for comparison

# Returns
- `true` if matrices are equal within tolerance, `false` otherwise
"""
function compare_matrices(expected::Matrix, actual::Matrix; tolerance::Float64=1e-10)
    if size(expected) != size(actual)
        @warn "Matrix sizes don't match: $(size(expected)) vs $(size(actual))"
        return false
    end
    
    return norm(expected - actual) < tolerance
end

"""
    test_conversion_correctness(convert_function::Function; verbose::Bool=true)

Test the conversion function with various test cases.

# Arguments
- `convert_function`: Function that takes (input_file, output_file) and converts symmetric to general
- `verbose`: Whether to print detailed test results

# Returns
- `true` if all tests pass, `false` otherwise
"""
function test_conversion_correctness(convert_function::Function; verbose::Bool=true)
    test_cases = [
        (:diagonal, "Diagonal matrix test"),
        (:mixed, "Mixed diagonal/off-diagonal test"), 
        (:full, "Full symmetric matrix test")
    ]
    
    all_passed = true
    
    if verbose
        println("="^60)
        println("TESTING SYMMETRIC TO GENERAL CONVERSION")
        println("="^60)
    end
    
    for (matrix_type, description) in test_cases
        if verbose
            println("\n🧪 $description")
            println("-"^40)
        end
        
        # Create test files
        input_file = "test_input_$(matrix_type).mtx"
        output_file = "test_output_$(matrix_type).mtx"
        
        try
            # Create test matrix
            expected_matrix, original_nnz = create_test_matrix(input_file, matrix_type)
            
            if verbose
                println("✓ Created test matrix ($(size(expected_matrix, 1))×$(size(expected_matrix, 2)), $original_nnz entries)")
            end
            
            # Convert using provided function
            convert_function(input_file, output_file)
            
            if verbose
                println("✓ Conversion completed")
            end
            
            # Parse converted matrix
            actual_matrix = parse_mtx_to_matrix(output_file)
            
            # Test 1: Size check
            if size(expected_matrix) != size(actual_matrix)
                if verbose
                    println("❌ Size mismatch: expected $(size(expected_matrix)), got $(size(actual_matrix))")
                end
                all_passed = false
                continue
            end
            
            # Test 2: Symmetry check
            if !verify_symmetry(actual_matrix)
                if verbose
                    println("❌ Converted matrix is not symmetric!")
                end
                all_passed = false
                continue
            end
            
            # Test 3: Content check
            if !compare_matrices(expected_matrix, actual_matrix)
                if verbose
                    println("❌ Matrix content doesn't match expected values")
                    println("Expected:")
                    display(expected_matrix)
                    println("Actual:")
                    display(actual_matrix)
                end
                all_passed = false
                continue
            end
            
            if verbose
                println("✅ All checks passed!")
                println("   - Size: $(size(actual_matrix))")
                println("   - Symmetry: ✓")
                println("   - Content: ✓")
            end
            
        catch e
            if verbose
                println("❌ Test failed with error: $e")
            end
            all_passed = false
        finally
            # Clean up test files
            isfile(input_file) && rm(input_file)
            isfile(output_file) && rm(output_file)
        end
    end
    
    if verbose
        println("\n" * "="^60)
        if all_passed
            println("🎉 ALL TESTS PASSED! Conversion is mathematically correct.")
        else
            println("❌ SOME TESTS FAILED! Check the conversion function.")
        end
        println("="^60)
    end
    
    return all_passed
end

end # module