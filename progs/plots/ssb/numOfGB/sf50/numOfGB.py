import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 13
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)
# plt.rcParams["font.family"] = "cursive"

# # # SSB - 5gb
# X = ["S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X = [0,1,2,3,4]
X_axis = np.arange(len(X)) 

bayesaqp_ssb_numpy_latency = [17.91333358269185, 66.5357667952776, 65.09020000230521,
                              71.27030820120126, 478.9780913852155] #[171.336,22.636,17.278,46.730,1988.690] 0 - 4
bayesaqp_ssb_torch_latency = [6.105216988362372, 31.2502249609679, 32.9460667911917,
                              38.81007500458509, 375.71996639017016]                           
deepdb_ssb_latency = [0.36671639999998900,
                        4.08822480000004,
                        4.965466800000050,
                        781.6252584000000,
                        2956.1010250000000] # [0.932,2.868,3.816,368.539,5666.250]

fig, ax = plt.subplots(figsize=(8,4))
# fig2, axs2 = plt.plots(1, figsize=(8,6))
# fig2.suptitle('Average Relative Error and Inference Latencies for the SSB (sf=5) dataset.')

deepdb_ssb_latency = np.array(deepdb_ssb_latency) / 1000
bayesaqp_ssb_numpy_latency = np.array(bayesaqp_ssb_numpy_latency) / 1000
bayesaqp_ssb_torch_latency = np.array(bayesaqp_ssb_torch_latency) / 1000
# inference latency
dfl_s = ax.bar(X_axis-0.2, deepdb_ssb_latency, 0.2, label="DeepDB", color='#377eb8')
bfln_s = ax.bar(X_axis, bayesaqp_ssb_numpy_latency, 0.2, label="BayesAQP-CPU", color='#ff7f00')
bflt_s = ax.bar(X_axis+0.2, bayesaqp_ssb_torch_latency, 0.2, label="BayesAQP-GPU", color='#dede00')


ax.set_ylabel("Inference Latency (sec)") 
# ax.set_yscale("log")
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0,1.02,1,0.15), borderaxespad=0, mode="expand")

plt.xticks(X_axis, X) 
# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig.savefig("progs/plots/ssb/numOfGB/sf50/ssb50gb_sigmod_numOfGB.pdf")