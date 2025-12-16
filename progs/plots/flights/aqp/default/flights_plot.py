import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# For comparing performance of AQP
import matplotlib

FONTSIZE = 17
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)

width = 0.27

X = ["F1.1", "F1.2", "F2.1", "F2.2", "F2.3", "F3.1", "F3.2", "F3.3", "F4.1", "F4.2", "F5.1", "F5.2"]
X_axis = np.arange(len(X)) 
# print(X)

bayesaqp_flights_error = [0.033,0.002,13.358,31.335,68.124,6.067,35.278,0.357,0.586,22.343,15.951,42.396]
deepdb_flights_error = [0.032,0.406,89.083,81.743,56.893,89.945,70.409,0.358,88.28,78.192,89.4,90.481]
verdictdb_flights_error = [0.052,0.554,22.399,97.623,54.762,16.807,165.374,1.002,24.846,60.064,38.614,61.033]

bayesaqp_flights_latency = [2.213,0.790,1.845,2.390,1.392,1.772,13.798,7.530,1.759,40.265,13.928,12.936]
deepdb_flights_latency = [2.199,1.515,1.937,2.56,2.897,9.513,11.945,12.481,24.938,50.491,49.098,24.284]
verdictdb_flights_latency = [33.999,29.294,37.155,34.963,26.445,30.638,32.237,27.249,37.402,31.156,26.945,36.987]

fig, axs = plt.subplots(2, figsize=(9,6.5), sharex=True, sharey=False)
# fig.suptitle('Average Relative Error and Inference Latencies for the Flights (default) dataset.')
plt.setp(axs, xticks=X_axis, xticklabels=X)
# relative error
bfe_s = axs[0].bar(X_axis - width, bayesaqp_flights_error, width, label = 'BayesAQP (ours)', color='#ff7f00') 
dfe_s = axs[0].bar(X_axis, deepdb_flights_error, width, label = 'DeepDB', color='#377eb8') 
vfe_s = axs[0].bar(X_axis + width, verdictdb_flights_error, width, label = 'VerdictDB', color='#4daf4a') 
# axs[0].bar_label(bfe_s, fmt='{:,.1f}%', rotation=90)
# axs[0].bar_label(dfe_s, fmt='{:,.1f}%', rotation=90)
# axs2[0].bar_label(vfe_s, fmt='{:,.1f}')
axs[0].set_ylabel("Relative Error (%)", size=FONTSIZE+2) 
axs[0].set_yscale("log")
# labels = [item for item in axs[0].get_yticklabels()]
# print(f"labels:{labels}")
axs[0].yaxis.grid(linestyle='--')
# axs[0].set_yticklabels(labels)
# axs[0].legend(loc='upper left')

count = 0 # annotate doesn't work with log scale
for bar in axs[0].patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        if count == 0:
            axs[0].text(x, height+1, f'Timeout', fontsize=15, weight='bold',
                ha='center', va='bottom', rotation=90)
        else: 
            axs[0].text(x, height+1, f'No result', fontsize=15, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    # else:
        # if height < 3:
        #     axs[0].text(x, height, f'{height:.1f}%', fontsize=16, weight='bold',
        #         ha='center', va='bottom', rotation=90)
            
# inference latency
# axs2[1].legend(loc='upper left')
bfl_s = axs[1].bar(X_axis - width, bayesaqp_flights_latency, width, label = 'BayesAQP(ours)', color='#ff7f00')
dfl_s = axs[1].bar(X_axis, deepdb_flights_latency, width, label = 'DeepDB', color='#377eb8') 
vfl_s = axs[1].bar(X_axis + width, verdictdb_flights_latency, width, label = 'VerdictDB', color='#4daf4a') 
# axs2[1].bar_label(bfl_s, fmt='{:,.1f}', rotation=90)
# axs2[1].bar_label(dfl_s, fmt='{:,.1f}', rotation=90)
# axs2[1].bar_label(vfl_s, fmt='{:,.1f}')
axs[1].set_ylabel("Inference Latency (ms)", size=FONTSIZE+2) 
axs[1].set_yscale("log")
axs[1].yaxis.grid(linestyle='--')
# axs[1].legend(loc='upper left')
axs[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0,1.02,1,0.08), borderaxespad=0, mode="expand")
count = 0 # annotate doesn't work with log scale
for bar in axs[1].patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        if count == 0:
            axs[1].text(x, height+17, f'Timeout', fontsize=15, weight='bold',
                ha='center', va='bottom', rotation=90)
        else: 
            axs[1].text(x, height+17, f'No result', fontsize=15, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    else:
        if height < 2:
            axs[1].text(x, height, f'{height:.1f}ms', fontsize=14.5, weight='bold',
                ha='center', va='bottom', rotation=90)
# plt.xticks(X_axis, X) 
# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig.align_ylabels(axs)
fig.savefig("progs/plots/flights_default_sigmod_aqp.pdf")