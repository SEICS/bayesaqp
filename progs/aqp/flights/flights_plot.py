import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# For comparing performance of AQP
# flights
bayesaqp_flights_path = "progs/aqp/flights/bayesaqp_compiler_warmup3_eval5.csv"
deepdb_flights_path = "progs/aqp/flights/deepdb_compiler_warmup3_eval5.csv"
bayesaqp_flights = pd.read_csv(bayesaqp_flights_path)
# print(bayesaqp_flights["query_no"])
# print(len(bayesaqp_flights))
deepdb_flights = pd.read_csv(deepdb_flights_path)
# print(deepdb_flights["query_no"])
flights_shared = list(set(bayesaqp_flights["query_no"]).intersection(set(deepdb_flights["query_no"])))
flights_shared.sort()
X = [f"Q{query_no+1}" for query_no in flights_shared]
X_axis = np.arange(len(X)) 
print(X)

bayesaqp_flights_error = bayesaqp_flights["relative_error(%)"]
deepdb_flights_error = deepdb_flights["average_relative_error"]
print(f"bayesaqp_flights_error: {bayesaqp_flights_error}")
print(f"deepdb_flights_error: {deepdb_flights_error}")
bayesaqp_flights_latency = bayesaqp_flights["latency_ms"]
deepdb_flights_latency = deepdb_flights["inference_latency_ms"]
print(f"bayesaqp_flights_latency: {bayesaqp_flights_latency}")
print(f"deepdb_flights_latency: {deepdb_flights_latency}")

fig, axs = plt.subplots(2, figsize=(12,6), sharex=True, sharey=False)
fig.suptitle('Average Relative Error and Inference Latencies for the Flights 1B dataset.')
plt.setp(axs, xticks=X_axis, xticklabels=X)
# relative error
# axs[0].legend(loc='upper left', ncols=2)
dfe_f = axs[0].bar(X_axis + 0.2, deepdb_flights_error, 0.4, label = 'DeepDB') 
bfe_f = axs[0].bar(X_axis - 0.2, bayesaqp_flights_error, 0.4, label = 'BayesAQP') 
axs[0].bar_label(dfe_f, fmt='{:,.1f}')
axs[0].bar_label(bfe_f, fmt='{:,.1f}')
# axs[0].set_xticks(X_axis, X) 
axs[0].set_ylabel("Average Relative Error (%)") 
axs[0].legend() 

# inference latency
# axs[1].legend(loc='upper left', ncols=2)
dfl_f = axs[1].bar(X_axis + 0.2, deepdb_flights_latency, 0.4, label = 'DeepDB') 
bfl_f = axs[1].bar(X_axis - 0.2, bayesaqp_flights_latency, 0.4, label = 'BayesAQP') 
axs[1].bar_label(dfl_f, fmt='{:,.1f}')
axs[1].bar_label(bfl_f, fmt='{:,.1f}')
# axs[1].set_xticks(X_axis, X) 
axs[1].set_ylabel("Average Inference Latency (ms)") 
axs[1].legend() 

# plt.xticks(X_axis, X) 
# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig.savefig("progs/aqp/flights/Flights_comparison.pdf")