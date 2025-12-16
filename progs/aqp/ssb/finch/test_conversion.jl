using SparseArrays
using LinearAlgebra

# Create a test symmetric matrix file
function create_test_symmetric_mtx(filename)
    open(filename, "w") do f
        println(f, "%%MatrixMarket matrix coordinate real symmetric")
        println(f, "%")
        println(f, "4 4 6")  # 4x4 matrix with 6 entries
        # Upper triangular entries only (symmetric format)
        println(f, "1 1 1.0")  # diagonal
        println(f, "1 2 2.0")  # off-diagonal
        println(f, "1 3 3.0")  # off-diagonal
        println(f, "2 2 4.0")  # diagonal
        println(f, "2 4 5.0")  # off-diagonal
        println(f, "3 3 6.0")  # diagonal
    end
end

# Test our conversion function
include("read_tensors.jl")

println("Creating test symmetric matrix...")
create_test_symmetric_mtx("test_symmetric.mtx")

println("\nOriginal file contents:")
println(read("test_symmetric.mtx", String))

println("Converting to general format...")
convert_symmetric_to_general("test_symmetric.mtx", "test_general.mtx")

println("\nConverted file contents:")
println(read("test_general.mtx", String))

println("Expected full matrix should be:")
println("1.0  2.0  3.0  0.0")
println("2.0  4.0  0.0  5.0") 
println("3.0  0.0  6.0  0.0")
println("0.0  5.0  0.0  0.0")

# Clean up
rm("test_symmetric.mtx")
rm("test_general.mtx")
