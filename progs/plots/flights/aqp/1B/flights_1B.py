from tkinter import W
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# For comparing performance of AQP
import matplotlib

FONTSIZE = 12
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)

width = 0.2

X = ["F1.1", "F1.2", "F2.1", "F2.2", "F2.3", "F3.1", "F3.2", "F3.3", "F4.1", "F4.2", "F5.1", "F5.2"]
X_axis = np.arange(len(X)) 

bayesaqp_flights_numpy_error = [0.0002,0.1833, 0.4048,0.6966,0.9595,0.3236,1.707,0.0167,3.8931,1.3474,0.4252,2.7492]
bayesaqp_flights_torch_error = [0.0002927658605014700,0.18331509563975700,0.40480691645471500,0.6965535187560800,0.9594811001281390,0.3434432642443170,2.505144944687270,0.016651405724834400,3.893113404516840,1.3474347769009000,0.42522048631725300,2.748371292279720]
deepdb_flights_error = [0.0003874383188733440,0.10521477714763300,0.3793185834681030,0.6783360903167800,0.9523189363650880,0.20560271115513900,1.846639365238110,0.01665533680369640,1.2301319787851000,0.6721307186493310,0.44365862592187600,42.16057997095870] # deepdb with python3.7 env 
verdictdb_flights_error = [0.005,0.28,2.21,8.331,10.86,1.722,9.394,0.277,4.241,9.256,3.427,49.392]
# RAM 16GB``
# bayesaqp_flights_latency = [3.598,0.903,2.295,2.095,1.806,0.545,3.744,5.460,1.197,10.748,4.346,4.213]
# deepdb_flights_latency = [1.107,0.595,0.778,1.077,1.027,3.987,5.29,6.346,8.617,15.891,10.298,4.71]
# verdictdb_flights_latency = [71.146,71.059,139.046,126.266,134.726,103.672,133.109,56.546,155.57,168.889,134.267,139.577] # need to figure out if this is really the inference latency for verdictdb

# RAM 48GB # NumPy-CPU, Torch-GPU
bayesaqp_flights_numpy_latency = [0.681, 0.274, 0.461, 0.673, 0.416, 0.473, 2.614, 1.757, 0.521, 2.199, 2.174, 3.177]
bayesaqp_flights_torch_latency = [1.469, 1.376, 2.419, 3.467, 2.686, 2.953, 4.171, 3.993, 3.992, 6.143, 4.412, 6.760]
# deepdb_flights_latency = [0.398,0.258,0.351,0.512,0.485,1.703,2.331,2.275,4.143,6.221,3.907,2.046]
deepdb_flights_latency = [0.68548319999997,0.39832480000001600,0.9834504000000080,4.74288319999998,1.453616400000010,3.089308000000020,4.962049800000010,4.421541799999980,133.41188320000000,286.3186000000000,10.747058200000100,4.9374666000002100] # deepdb with python3.7 env 
verdictdb_flights_latency = [82.778,72.033,152.064,134.11,134.052,115.556,145.224,66.399,155.41,174.066,144.049,157.468]

fig, axs = plt.subplots(2, figsize=(9,6.5), sharex=False, sharey=False)

# relative error
vfe_s = axs[0].bar(X_axis - 1.5*width , verdictdb_flights_error, width, label = 'VerdictDB', color='#4daf4a') 
dfe_s = axs[0].bar(X_axis - 0.5*width, deepdb_flights_error, width, label = 'DeepDB', color='#377eb8') 
bfne_s = axs[0].bar(X_axis + 0.5*width, bayesaqp_flights_numpy_error, width, label = 'BayesAQP-CPU', color='#ff7f00') 
bfte_s = axs[0].bar(X_axis + 1.5*width, bayesaqp_flights_torch_error, width, label = 'BayesAQP-GPU', color='#dede00') 

axs[0].set_ylabel("Relative Error (%)", size=FONTSIZE+2) 
# axs[0].set_yscale("log")
axs[0].yaxis.grid(linestyle='--')

count = 0 # annotate doesn't work with log scale
for bar in axs[0].patches:
    height = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        if count == 0:
            axs[0].text(x, height+1, f'Timeout', fontsize=10, weight='bold',
                ha='center', va='bottom', rotation=90)
        else: 
            axs[0].text(x, height+1, f'No result', fontsize=10, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    else:
        if height < 2:
            axs[0].text(x, height, f'{height:.1f}%', fontsize=10, weight='bold',
                ha='center', va='bottom', rotation=90)
            
# inference latency
vfl_s = axs[1].bar(X_axis - 1.5*width, verdictdb_flights_latency, width, label = 'VerdictDB', color='#4daf4a') 
dfl_s = axs[1].bar(X_axis - 0.5*width, deepdb_flights_latency, width, label = 'DeepDB', color='#377eb8')
bfnl_s = axs[1].bar(X_axis + 0.5*width, bayesaqp_flights_numpy_latency, width, label = 'BayesAQP-CPU', color='#ff7f00')
bftl_s = axs[1].bar(X_axis + 1.5*width, bayesaqp_flights_torch_latency, width, label = 'BayesAQP-GPU', color='#dede00')

# vfl_s = axs[1].bar(X_axis + width, verdictdb_flights_latency, width, label = 'VerdictDB', color='#4daf4a') 
axs[1].set_ylabel("Inference Latency (ms)", size=FONTSIZE+2) 
axs[1].set_yscale("log")
axs[1].yaxis.grid(linestyle='--')
count = 0 # annotate doesn't work with log scale
for bar in axs[1].patches:
    height = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        if count == 0:
            axs[1].text(x, height+17, f'Timeout', fontsize=10, weight='bold',
                ha='center', va='bottom', rotation=90)
        else: 
            axs[1].text(x, height+17, f'No result', fontsize=10, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    # else:
    #     if height < 2:
    #         axs[1].text(x, height, f'{height:.1f}ms', fontsize=10, weight='bold',
    #             ha='center', va='bottom', rotation=90)

axs[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0,1.02,1,0.08), borderaxespad=0, mode="expand")
plt.setp(axs, xticks=X_axis, xticklabels=X) # one xticklabel for 2 subplots
axs[0].set_xlabel("Query")
axs[1].set_xlabel("Query")
plt.tight_layout()
fig.align_ylabels(axs)
fig.savefig("progs/plots/flights/aqp/1B/flights_1B_sigmod_aqp.pdf")