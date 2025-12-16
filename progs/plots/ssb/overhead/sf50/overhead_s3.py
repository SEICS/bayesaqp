import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns

# Set style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FONTSIZE = 17
matplotlib.rc('font', size=FONTSIZE, family='DejaVu Sans')
matplotlib.rc('axes', titlesize=FONTSIZE+2)
matplotlib.rc('axes', labelsize=FONTSIZE+1)
matplotlib.rc('legend', fontsize=FONTSIZE-1)
matplotlib.rc('xtick', labelsize=FONTSIZE-1)
matplotlib.rc('ytick', labelsize=FONTSIZE-1)

# Define a professional color palette
colors_bayesaqp = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E', '#577590']
colors_verdictdb = ['#FF6B6B']

X = ["S1.1", "S1.2", "S1.3", "S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]

e2e = { # percentage
    'parse': np.array([60.4, 60.2, 61, 77.8, 74, 79.2, 83.6, 84.2, 84.8, 84.4, 88.8, 91, 74.0]),
    'TensorLowering': np.array([26.8, 26.6, 26, 36.6, 35.8, 32.8, 33.8, 34.4, 32.8, 33, 40.4, 41.4, 44.6]),
    'TensorToSetar': np.array([8, 9.4, 9.8, 9, 9.8, 10.2, 10.6, 10.8, 11, 13.2, 13.2, 13.2, 14.6]),
    'O1': np.array([13.4, 13.2, 13.6, 19.2, 20, 22.2, 24.6, 23.6, 23.4, 23.8, 28, 29.6, 30.2]),
    'O2': np.array([3.6, 4, 4, 6.6, 5.4, 4.4, 5, 5, 5, 5, 6.8, 6.4, 5.8]),
    'SetarToTensor': np.array([3.4, 3.8, 4, 3, 3, 3.4, 3, 3, 3, 4, 3.2, 3.4]),
    'Codegen': np.array([25.2, 25.8, 26.4, 25.6, 29, 25.8, 25.2, 24.8, 24.4, 24.8, 29.4, 30.2, 33.8]),
    'Python Overhead': np.array([13.667833389155566, 13.237541563622653, 13.194983218610286, 14.742141606286168, 13.966308234259486, 14.0672836413607, 13.762242025323213, 14.173333216644824, 13.86125001590699, 14.093991583772004, 14.05025820620358, 13.820308417081833, 14.257883416675032]),
    'Inference': np.array([2.653, 2.533, 2.738, 16.133, 6.697, 4.899, 10.223, 32.694, 9.837, 9.757, 10.154, 18.371, 92.330])
}

verdictdb = {
    'Preprocessing (VerdictDB)' : np.array([11.250, 6.683, 5.754, 53.185, 23.121, 10.063, 43.967, 111.028, 14.248, 0, 18.2080, 36.702, 50.096]),
    'DB execution (VerdictDB)' : np.array([48.6, 40.8, 46.8, 426.6, 488.4, 247.8, 528.6, 316.4, 204.4, 0, 718.2, 535.4, 387.4])
}

# Process data
cmp_ovh = list(zip(*e2e.values()))
cmp_ovh_q = dict(zip(X, cmp_ovh))
d = cmp_ovh_q

X2 = list(d.keys())
X_axis = np.arange(len(X2)) 
e2e_processed = list(zip(*d.values()))

# Filter for S3 queries only
s3_queries = [q for q in X2 if q.startswith('S3')]
s3_indices = [X2.index(q) for q in s3_queries]

# Create processed data structures
inference_dict = dict(zip(X, e2e_processed[-1]))
e2e_final = {
    'IR Translation (BayesAQP)': np.array(e2e_processed[1]) + np.array(e2e_processed[2]) + np.array(e2e_processed[5]),
    'O1 (BayesAQP)': np.array(e2e_processed[3]),
    'O2 (BayesAQP)': np.array(e2e_processed[4]),
    'Codegen (BayesAQP)': np.array(e2e_processed[6]),
    'Python Overhead (BayesAQP)': np.array(e2e_processed[7])
}

inference_sorted = {key: inference_dict[key] for key in X2}
inference = {
    'Inference (BayesAQP)': np.array(list(inference_sorted.values()))
}

verdictdb_ovh = list(zip(*verdictdb.values()))
verdictdb_ovh_q = dict(zip(X, verdictdb_ovh))
verdictdb_sort = {key: verdictdb_ovh_q[key] for key in X2}
verdictdb_values = list(zip(*verdictdb_sort.values()))
verdictdb_final = {
    'Preprocessing (VerdictDB)' : np.array(verdictdb_values[0]),
    'DB execution (VerdictDB)' : np.array(verdictdb_values[1])
}

# Create single plot for S3 queries
width = 0.7
fig, ax = plt.subplots(figsize=(12, 8))  # Slightly wider for 4 queries
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

X_axis_group = np.arange(len(s3_queries))
bottom = np.zeros(len(s3_queries))
compilation_bottom = np.zeros(len(s3_queries))

# Plot BayesAQP components
count = 0
for phase, e2e_data in e2e_final.items():
    group_data = e2e_data[s3_indices]
    color = colors_bayesaqp[count % len(colors_bayesaqp)]
    p = ax.bar(X_axis_group-width/4, group_data, width/2, label=phase, 
               bottom=bottom, color=color, edgecolor='white', linewidth=0.5, alpha=0.9)
    bottom += group_data
    compilation_bottom += group_data
    
    ax.bar_label(p, fmt='%.0f', label_type='center', fontsize=12, weight='bold', color='black')
    count += 1

# Add inference layer
for phase, inf_data in inference.items():
    group_inf_data = inf_data[s3_indices]
    color = '#0072B2'
    p_inf = ax.bar(X_axis_group-width/4, group_inf_data, width/2, 
                   label=phase, bottom=bottom, color=color, edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.bar_label(p_inf, fmt='%.1f', label_type='center', fontsize=12, weight='bold', color='black')

# Add VerdictDB components
bottom_vdb = np.zeros(len(s3_queries))
for phase, vdb in verdictdb_final.items():
    group_vdb_data = vdb[s3_indices]
    if 'Preprocessing' in phase:
        color = colors_verdictdb[0]
    else:
        color = '#FF9999'
    
    p2 = ax.bar(X_axis_group+width/4, group_vdb_data, width/2, 
                label=phase, bottom=bottom_vdb, color=color, edgecolor='white', linewidth=0.5, alpha=0.9)
    bottom_vdb += group_vdb_data
    ax.bar_label(p2, fmt=lambda x: f'{x:.0f}' if x > 0 else '', label_type='center', 
                 fontsize=12, weight='bold', color='black')

# Add mean compilation line
mean_compile_group = np.mean(compilation_bottom)
ax.axhline(mean_compile_group, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

# Customize plot
ax.set_title('S3 Queries: BayesAQP vs VerdictDB Overhead Comparison', fontsize=FONTSIZE+3, weight='bold', pad=20)
ax.set_yscale('log')
ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray', linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xticks(X_axis_group)
ax.set_xticklabels(s3_queries, fontsize=FONTSIZE+1)
ax.set_xlabel("Queries", size=FONTSIZE+2, weight='bold')
ax.set_ylabel("Compilation Overhead Time (ms)", size=FONTSIZE+1, weight='bold')
ax.tick_params(axis='y', labelsize=FONTSIZE)

# Add legend - expanded to full width
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0, 1.02, 1, 0.1), 
          frameon=True, fancybox=True, shadow=False, facecolor='white', 
          prop={'size': FONTSIZE-1}, columnspacing=1.5, 
          handletextpad=0.8, handlelength=1.5, mode="expand", borderaxespad=0)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)

# Save plot
fig.savefig("progs/plots/ssb/overhead/sf50/ssb50gb_overhead_s3.pdf", 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

# Optionally show the plot
# plt.show()
