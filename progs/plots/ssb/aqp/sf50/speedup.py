import numpy as np
bayesaqp_ssb_numpy_latency = [2.653, 2.533, 2.738, 16.133, 6.697, 4.899, 10.223, 32.694, 9.837, 9.757, 10.154, 18.371, 92.330]
bayesaqp_ssb_torch_latency = [4.599, 4.546, 5.340, 12.986, 10.142, 8.902, 12.965, 27.681, 13.148, 12.459, 13.877, 18.223, 49.297]

deepdb_ssb_latency = [3.836324999999970,3.717583400000060,3.7750082000000500,34.552574999999900,12.059274600000000,6.856532999999980,20.979400000000000,67.14849160000010,8.642575000000190,10.169874800000000,15.514299999999900,29.08630820000000,174.22420820000000] # deepdb with python3.7 env 
verdictdb_ssb_latency = [48.6,40.8,46.8,426.6,488.4,247.8,528.6,316.4,204.4,0,718.2,535.4,387.4]

bayesaqp_ssb_numpy_latency = np.array(bayesaqp_ssb_numpy_latency)
bayesaqp_ssb_torch_latency = np.array(bayesaqp_ssb_torch_latency)
deepdb_ssb_latency = np.array(deepdb_ssb_latency)
verdictdb_ssb_latency = np.array(verdictdb_ssb_latency)

sp1 = deepdb_ssb_latency / bayesaqp_ssb_numpy_latency
sp2 = deepdb_ssb_latency / bayesaqp_ssb_torch_latency
sp3 = verdictdb_ssb_latency / bayesaqp_ssb_numpy_latency
sp4 = verdictdb_ssb_latency / bayesaqp_ssb_torch_latency
print("average DeepDB / BayesAQP-CPU: ", np.mean(sp1))
print("average DeepDB / BayesAQP-GPU: ", np.mean(sp2))
print("average VerdictDB / BayesAQP-CPU: ", np.mean(sp3))
print("average VerdictDB / BayesAQP-GPU: ", np.mean(sp4))
print("average SSB sf50 speedup: ", np.mean(np.concatenate([sp1, sp2, sp3, sp4])))
print("\n")
print("max DeepDB / BayesAQP-CPU: ", max(sp1))
print("max DeepDB / BayesAQP-GPU: ", max(sp2))
print("max VerdictDB / BayesAQP-CPU: ", max(sp3))
print("max VerdictDB / BayesAQP-GPU: ", max(sp4))
print("max SSB sf50 speedup: ", max(np.concatenate([sp1, sp2, sp3, sp4])))
print("\n")
# Flights 1B
bayesaqp_flights_numpy_latency = [0.681, 0.274, 0.461, 0.673, 0.416, 0.473, 2.614, 1.757, 0.521, 2.199, 2.174, 3.177]
bayesaqp_flights_torch_latency = [1.469, 1.376, 2.419, 3.467, 2.686, 2.953, 4.171, 3.993, 3.992, 6.143, 4.412, 6.760]
deepdb_flights_latency = [0.68548319999997,0.39832480000001600,0.9834504000000080,4.74288319999998,1.453616400000010,3.089308000000020,4.962049800000010,4.421541799999980,133.41188320000000,286.3186000000000,10.747058200000100,4.9374666000002100] # deepdb with python3.7 env 
verdictdb_flights_latency = [82.778,72.033,152.064,134.11,134.052,115.556,145.224,66.399,155.41,174.066,144.049,157.468]

bayesaqp_flights_numpy_latency = np.array(bayesaqp_flights_numpy_latency)
bayesaqp_flights_torch_latency = np.array(bayesaqp_flights_torch_latency)
deepdb_flights_latency = np.array(deepdb_flights_latency)
verdictdb_flights_latency = np.array(verdictdb_flights_latency)

sp5 = deepdb_flights_latency / bayesaqp_flights_numpy_latency
sp6 = deepdb_flights_latency / bayesaqp_flights_torch_latency
sp7 = verdictdb_flights_latency / bayesaqp_flights_numpy_latency
sp8 = verdictdb_flights_latency / bayesaqp_flights_torch_latency
print("average DeepDB / BayesAQP-CPU: ", np.mean(sp5))
print("average DeepDB / BayesAQP-GPU: ", np.mean(sp6))
print("average VerdictDB / BayesAQP-CPU: ", np.mean(sp7))
print("average VerdictDB / BayesAQP-GPU: ", np.mean(sp8))
print("average Flights 1B speedup: ", np.mean(np.concatenate([sp5, sp6, sp7, sp8])))
print("\n")
print("max DeepDB / BayesAQP-CPU: ", max(sp5))
print("max DeepDB / BayesAQP-GPU: ", max(sp6))
print("max VerdictDB / BayesAQP-CPU: ", max(sp7))
print("max VerdictDB / BayesAQP-GPU: ", max(sp8))
print("max Flights 1B speedup: ", max(np.concatenate([sp5, sp6, sp7, sp8])))
print("\n")
print("average speedup: ", np.mean(np.concatenate([sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8])))
print("max speedup: ", max(np.concatenate([sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8])))
print("\n")

# accuracy
bayesaqp_flights_numpy_error = [0.0002,0.1833, 0.4048,0.6966,0.9595,0.3236,1.707,0.0167,3.8931,1.3474,0.4252,2.7492]
bayesaqp_flights_torch_error = [0.0002927658605014700,0.18331509563975700,0.40480691645471500,0.6965535187560800,0.9594811001281390,0.3434432642443170,2.505144944687270,0.016651405724834400,3.893113404516840,1.3474347769009000,0.42522048631725300,2.748371292279720]
print("average BayesAQP-CPU Flights 1B error: ", np.mean(bayesaqp_flights_numpy_error))
print("average BayesAQP-GPU Flights 1B error: ", np.mean(bayesaqp_flights_torch_error))
print("average Flights 1B error: ", np.mean(np.concatenate([bayesaqp_flights_numpy_error, bayesaqp_flights_torch_error])))


bayesaqp_ssb_numpy_error = [3.156413507233165, 2.0804292663749413, 1.8218650696177674, 3.3160402392906003,
                       1.9195032167735573, 1.068065304133672, 0.5215648939048564, 3.563405931913926,
                        3.3059749786573196, 6.960586331149478, 0.4340643096952319, 0.9549783061120759,
                        19.525864995443133]
bayesaqp_ssb_torch_error = [3.156406463417058, 2.0804223610852643, 1.8218747798329813,  3.3160395572608206,
                            1.91950044109579, 1.0680634615118343, 0.5215606583968653, 3.5634080822839054,
                            3.3059939546908335, 6.960583462738758, 0.4340614734233384, 0.9549839529443702,
                            19.525874558490884]
deepdb_ssb_error = [0.23349619232903600,0.3075223594522370,0.07641302360895430,
                    2.4049298501062500,1.6274797373812000,1.7811574204956500,
                    0.5697948074716060,3.950736891629180,3.999654979664840,
                    10.097852661686000,0.3844116824749660,0.9669633510492590,
                    19.30403830246310]
verdictdb_ssb_error = [0.929,2.73,1.83,10.809,
                        9.149,10.661,3.465,33.72,
                        38.953,0,2.06,8.248,
                        311.498]
bayesaqp_ssb_numpy_error = np.array(bayesaqp_ssb_numpy_error)
bayesaqp_ssb_torch_error = np.array(bayesaqp_ssb_torch_error)
deepdb_ssb_error = np.array(deepdb_ssb_error)
verdictdb_ssb_error = np.array(verdictdb_ssb_error)
print("average BayesAQP-CPU SSB error: ", np.mean(bayesaqp_ssb_numpy_error))
print("average BayesAQP-GPU SSB error: ", np.mean(bayesaqp_ssb_torch_error))
temp = deepdb_ssb_error - bayesaqp_ssb_numpy_error
print("mean diff DeepDB error vs BayesAQP-CPU: ", np.mean(abs(temp)))
print("max diff DeepDB error vs BayesAQP-CPU: ", list(temp).index(max(temp)))