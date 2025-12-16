import numpy as np
bayesaqp_flights_numpy_latency = [0.681, 0.274, 0.461, 0.673, 0.416, 0.473, 2.614, 1.757, 0.521, 2.199, 2.174, 3.177]
bayesaqp_flights_torch_latency = [1.469, 1.376, 2.419, 3.467, 2.686, 2.953, 4.171, 3.993, 3.992, 6.143, 4.412, 6.760]
deepdb_flights_latency = [0.68548319999997,0.39832480000001600,0.9834504000000080,4.74288319999998,1.453616400000010,3.089308000000020,4.962049800000010,4.421541799999980,133.41188320000000,286.3186000000000,10.747058200000100,4.9374666000002100] # deepdb with python3.7 env 
verdictdb_flights_latency = [82.778,72.033,152.064,134.11,134.052,115.556,145.224,66.399,155.41,174.066,144.049,157.468]

bayesaqp_flights_numpy_latency = np.array(bayesaqp_flights_numpy_latency)
bayesaqp_flights_torch_latency = np.array(bayesaqp_flights_torch_latency)
deepdb_flights_latency = np.array(deepdb_flights_latency)
verdictdb_flights_latency = np.array(verdictdb_flights_latency)

sp1 = deepdb_flights_latency / bayesaqp_flights_numpy_latency
sp2 = deepdb_flights_latency / bayesaqp_flights_torch_latency
sp3 = verdictdb_flights_latency / bayesaqp_flights_numpy_latency
sp4 = verdictdb_flights_latency / bayesaqp_flights_torch_latency
print("DeepDB / BayesAQP-CPU: ", np.mean(sp1))
print("DeepDB / BayesAQP-GPU: ", np.mean(sp2))
print("VerdictDB / BayesAQP-CPU: ", np.mean(sp3))
print("VerdictDB / BayesAQP-GPU: ", np.mean(sp4))
print("average speedup: ", np.mean(np.concatenate([sp1, sp2, sp3, sp4])))