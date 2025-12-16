import numpy as np
bayesaqp_ssb_numpy_latency = [66.46740820724517, 67.29318322613835, 68.738008197397,
                                69.91482498124242, 71.06670020148158, 72.85699183121324,
                                74.10914998035878, 75.20444998517632] # 74.89827498793602
bayesaqp_ssb_torch_latency = [33.96268361248076,33.52811641525477,35.9076748136431,
                                39.26915819756687, 40.40913339704275, 39.6605916088447,
                                41.209674812853336, 41.42360838595778] #42.02754981815815
deepdb_ssb_latency = [10.390966600000000, 15.286891600000000, 22.468283400000000,
                        30.428050000000000, 44.575941800000000, 51.4914166,
                        59.314025200000000, 68.08642500000000]

bayesaqp_ssb_numpy_latency = np.array(bayesaqp_ssb_numpy_latency)
bayesaqp_ssb_torch_latency = np.array(bayesaqp_ssb_torch_latency)
deepdb_ssb_latency = np.array(deepdb_ssb_latency)

sp1 = deepdb_ssb_latency / bayesaqp_ssb_numpy_latency
sp2 = deepdb_ssb_latency / bayesaqp_ssb_torch_latency
print("average DeepDB / BayesAQP-CPU speedup: ", np.mean(sp1))
print("average DeepDB / BayesAQP-GPU speedup: ", np.mean(sp2[2:]))
print("average speedup: ", np.mean(np.concatenate([sp1, sp2])))
print("\n")
print("max DeepDB / BayesAQP-CPU speedup: ", max(sp1))
print("max DeepDB / BayesAQP-GPU speedup: ", max(sp2))
print("max speedup: ", max(np.concatenate([sp1, sp2])))