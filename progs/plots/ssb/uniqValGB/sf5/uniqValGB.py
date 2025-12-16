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
X = [1000,8372] #[5,25,150,1000]
X_axis = np.arange(len(X)) 

# bayesaqp_ssb_latency = [23.274,42.124,76.836,315.276] # p_mfgr, p_categorty, p_type, p_brand1, p_name
# deepdb_ssb_latency = [3.547,3.542,4.633,18.268]

bayesaqp_ssb_latency = [19.392,75.773] # p_mfgr, p_categorty, p_type, p_brand1, p_name
deepdb_ssb_latency = [17.057,142.245]

fig2, axs2 = plt.subplots(figsize=(8,4))
# fig2.suptitle('Average Relative Error and Inference Latencies for the SSB (sf=5) dataset.')

# bfe_s = axs2[0].bar(X_axis, bayesaqp_ssb_error, 0.2, color='#ff7f00') 

# axs2[0].set_ylabel("Relative Error (%)") 
# axs2[0].set_yscale("log")
# axs2[0].legend(loc='upper left')

# count = 0 # annotate doesn't work with log scale
# for bar in axs2[0].patches:
#     height = bar.get_height()
#     # print(f"height == 0: {height == 0}")
#     x = bar.get_x() + bar.get_width() / 2.0
#     if height == 0:
#         # print(f"count: {count}")
#         if count == 0:
#             axs2[0].text(x, height+1, f'Timeout', fontsize=16, weight='bold',
#                 ha='center', va='bottom', rotation=90)
#         else: 
#             axs2[0].text(x, height+1, f'No result', fontsize=16, weight='bold',
#                 ha='center', va='bottom', rotation=90)
#         count += 1
#     else:
#         if height < 2:
#             axs2[0].text(x, height, f'{height:.1f}%', fontsize=16, weight='bold',
#                 ha='center', va='bottom', rotation=90)
    
# inference latency
bfl_s = axs2.bar(X_axis-0.1, bayesaqp_ssb_latency, 0.2, label="BayesAQP (ours)", color='#ff7f00')
dfl_s = axs2.bar(X_axis+0.1, deepdb_ssb_latency, 0.2, label="DeepDB", color='#377eb8')

axs2.set_ylabel("Inference Latency (ms)") 
axs2.set_yscale("log")
# axs2[1].legend(loc='upper left')
# axs2[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0,1.02,1,0.08), borderaxespad=0, mode="expand")
# count = 0 # annotate doesn't work with log scale
# for bar in axs2[1].patches:
#     height = bar.get_height()
#     # print(f"height == 0: {height == 0}")
#     x = bar.get_x() + bar.get_width() / 2.0
#     if height == 0:
#         # print(f"count: {count}")
#         if count == 0:
#             axs2[1].text(x, height+17, f'Timeout', fontsize=16, weight='bold',
#                 ha='center', va='bottom', rotation=90)
#         else: 
#             axs2[1].text(x, height+17, f'No result', fontsize=16, weight='bold',
#                 ha='center', va='bottom', rotation=90)
#         count += 1
#     else:
#         if height < 3:
#             axs2[1].text(x, height, f'{height:.1f}%', fontsize=16, weight='bold',
#                 ha='center', va='bottom', rotation=90)

plt.xticks(X_axis, X) 
axs2.legend(ncol=2, loc="upper left", bbox_to_anchor=(0,1.02,1,0.18), borderaxespad=0, mode="expand")

# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig2.savefig("plots/ssb5gb_sigmod_uniqValGB.pdf")