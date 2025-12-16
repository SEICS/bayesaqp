import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 17
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)
# plt.rcParams["font.family"] = "cursive"

# # # SSB - 5gb
# X = ["S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X = [2,3,4]
X_axis = np.arange(len(X)) 

bayesaqp_ssb_latency = [17.278,46.730,1988.690] #[171.336,22.636,17.278,46.730,1988.690] 0 - 4

deepdb_ssb_latency = [3.816,368.539,5666.250] # [0.932,2.868,3.816,368.539,5666.250]

fig, ax = plt.subplots(figsize=(8,4))
# fig2, axs2 = plt.plots(1, figsize=(8,6))
# fig2.suptitle('Average Relative Error and Inference Latencies for the SSB (sf=5) dataset.')

# inference latency
bfl_s = ax.bar(X_axis-0.1, bayesaqp_ssb_latency, 0.2, label="BayesAQP (ours)", color='#ff7f00')
dfl_s = ax.bar(X_axis+0.1, deepdb_ssb_latency, 0.2, label="DeepDB", color='#377eb8')

ax.set_ylabel("Inference Latency (ms)") 
ax.set_yscale("log")
ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0,1.02,1,0.15), borderaxespad=0, mode="expand")

plt.xticks(X_axis, X) 
# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig.savefig("plots/ssb5gb_sigmod_numOfGB.pdf")