using Finch, BenchmarkTools;

# julia> fsprand(Float64, 2, 2, 2, 0.5)
# SparseCOO (0.0) [1:2,1:2,1:2]
# ├─├─├─[2, 2, 1]: 0.6478553157718558
# ├─├─├─[1, 1, 2]: 0.996665291437684
# ├─├─├─[2, 1, 2]: 0.7491940599574348
A = fsprand(1000, 1000, 0.1); # shape, shape, sparsity
B = fsprand(1000, 1000, 0.1);
C = fsprand(1000, 1000, 0.0001);

A = lazy(A); # lazy evaluation
B = lazy(B);
C = lazy(C);

# Note: Finch doesn't support integer index labels but supporting abstract index labels. It also supports character index labels by default.
# For example, `@einsum R[a1,a4] = A[a1,a2] * B[a2,a3] * C[a3,a4]` is valid,
# but `@einsum R[1,4] = A[1,2] * B[2,3] * C[3,4]` is not.

# Run einsum with Galley scheduler, and define the benchmark.
b = @benchmarkable(
    begin
        @einsum R[a1,a4] = A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4]
    end,
    setup = Finch.set_scheduler!(Finch.galley_scheduler());
)

tb = tune!(b);

@show run(tb);

@btime begin
    @einsum R[a1,a4] = A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4]
    compute(R)
end

# Run einsum with Galley scheduler but measure the overall execution time.
@btime Finch.with_scheduler(Finch.galley_scheduler()) do
    @einsum R[a1,a4] = A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4] * A[a1,a2] * B[a2,a3] * C[a3,a4]
    compute(R)
end
