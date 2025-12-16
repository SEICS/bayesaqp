include("finch_load_tensors.jl") # load .npz and .mtx files and convert tensors to Finch.Tensor
include("reverse_map.jl") # reverse map function that converts discretized tensor results back to original group-by keys
include("ssb/ssb_sf50_q3.jl")

using BenchmarkTools

function update_global_variables(vars)
    for (k, v) in vars
        sym = Symbol(k)
        if isa(v, Tensor) && !endswith(string(k), "_b2v")
            @eval const $sym = lazy($v) 
        else
            @eval const $sym = $v 
        end
    end
end

function main()
    cpds_folder = "progs/aqp/ssb/finch/data"
    cpds = finch_load_all(cpds_folder)
    update_global_variables(cpds)
    
    Finch.set_scheduler!(Finch.galley_scheduler())
    q3 = @btime ssb_sf50_q3(reverse_map, true)
    println(size(q3))
end

main()