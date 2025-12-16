import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 17
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)

# # # SSB - 5gb
X = ["S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X_axis = np.arange(len(X)) 

dp = [64.3198384000016,
26.783301200021,
21.3875056000234,
46.1069563999445,
142.064021999977,
31.6860601999906,
29.7971963999771,
51.7050286000085,
104.852731400024,
385.365180400095]

greedy = [441.86557899975,
302.0832866001,
295.436929399875,
151.868321000075,
213.193507400138,
128.227491400139,
126.556233199881,
529.742499000167,
617.745260599986,
825.126506600282]

# random-greedy 32
rg32 = [295.800660000077,
125.719695400039,
122.372497199831,
79.3928791998042,
174.447660999977,
83.5507317999145,
70.3347685999688,
403.135682600077,
456.818517000283,
643.716979399869]

rg128 = [355.522424399896,
219.997469999907,
211.569111199969,
192.262940999899,
264.873326599809,
163.472206400002,
177.348462199734,
506.473530600124,
537.936689399794,
812.357952599996]
 
fig, axs = plt.subplots(figsize=(8,4))
# fig.suptitle('Inference Latencies of BayesAQP in AQP using Different Contraction Strategies for SSB (sf=5) dataset.')

# inference latency
dpl_s= axs.bar(X_axis - 0.3, dp, 0.2, label = 'DP', color='#ff7f00')
gl_s = axs.bar(X_axis-0.1, greedy, 0.2, label = 'G1', color='#377eb8') 
rg32_s = axs.bar(X_axis+0.1, rg32, 0.2, label = 'G32', color='#4daf4a') 
rg128_s = axs.bar(X_axis + 0.3, rg128, 0.2, label = 'G128', color='#dede00') 
# axs.bar_label(bfl_s, fmt='{:,.1f}', rotation=90)
# axs.bar_label(dfl_s, fmt='{:,.1f}', rotation=90)
# axs.bar_label(vfl_s, rotation=90, labels = ["Timeout", "", "", "", "", "", "No result", "", "", "No result"])
axs.set_ylabel("Inference Latency (ms)", size=FONTSIZE+2) 
axs.set_yscale("log")
legend = axs.legend(ncol=4, loc="upper left", bbox_to_anchor=(0,1.02,1,0.16), borderaxespad=0, mode="expand")
plt.setp(legend.get_title(),fontsize=15)
# count = 0 # annotate doesn't work with log scale
# for bar in axs.patches:
#     height = bar.get_height()
#     # print(f"height == 0: {height == 0}")
#     x = bar.get_x() + bar.get_width() / 2.0
#     if height == 0:
#         # print(f"count: {count}")
#         if count == 0:
#             axs.text(x, height+17, f'Timeout', fontsize=15, weight='bold',
#                 ha='center', va='bottom', rotation=90)
#         else: 
#             axs.text(x, height+17, f'No result', fontsize=15, weight='bold',
#                 ha='center', va='bottom', rotation=90)
#         count += 1
#     else:
#         if height < 3:
#             axs.text(x, height, f'{height:.2f}%', fontsize=15, weight='bold',
#                 ha='center', va='bottom', rotation=90)

plt.xticks(X_axis, X) 
# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig.savefig("plots/ssb5gb_contract_strategy.pdf")