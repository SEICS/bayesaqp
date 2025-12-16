import numpy as np

bayesaqp_ssb_numpy_latency = [17.91333358269185, 66.5357667952776, 65.09020000230521,
                              71.27030820120126, 478.9780913852155] #[171.336,22.636,17.278,46.730,1988.690] 0 - 4
bayesaqp_ssb_torch_latency = [6.105216988362372, 31.2502249609679, 32.9460667911917,
                              38.81007500458509, 375.71996639017016]                           
deepdb_ssb_latency = [0.36671639999998900,
                        4.08822480000004,
                        4.965466800000050,
                        781.6252584000000,
                        2956.1010250000000] 

bayesaqp_ssb_numpy_latency = np.array(bayesaqp_ssb_numpy_latency)
bayesaqp_ssb_torch_latency = np.array(bayesaqp_ssb_torch_latency)
deepdb_ssb_latency = np.array(deepdb_ssb_latency)

sp1 = deepdb_ssb_latency / bayesaqp_ssb_numpy_latency
sp2 = deepdb_ssb_latency / bayesaqp_ssb_torch_latency
print("average DeepDB / BayesAQP-CPU speedup: ", np.mean(sp1))
print("average DeepDB / BayesAQP-GPU speedup: ", np.mean(sp2))
print("average SSB sf50 speedup: ", np.mean(np.concatenate([sp1, sp2])))
print("\n")
print("max DeepDB / BayesAQP-CPU speedup: ", max(sp1))