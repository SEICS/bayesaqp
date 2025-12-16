import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns

# Set modern, professional style
plt.style.use('default')
sns.set_palette("Set2")

# Enhanced font settings for larger figure
FONTSIZE = 20
matplotlib.rc('font', size=FONTSIZE, family='Arial', weight='normal')
matplotlib.rc('axes', titlesize=FONTSIZE+4)
matplotlib.rc('axes', labelsize=FONTSIZE+2)
matplotlib.rc('legend', fontsize=FONTSIZE)
matplotlib.rc('xtick', labelsize=FONTSIZE)
matplotlib.rc('ytick', labelsize=FONTSIZE)
matplotlib.rc('figure', facecolor='white')
matplotlib.rc('axes', facecolor='white')
matplotlib.rc('axes', edgecolor='#2c3e50')
matplotlib.rc('axes', linewidth=1.5)

# Define colorblind-friendly palette (based on Cynthia Brewer's ColorBrewer)
colors_bayesaqp = [
    '#d62728',  # Red (high contrast) - swapped from position 4
    '#ff7f0e',  # Orange (distinguishable from blue)
    '#2ca02c',  # Green (safe green)
    '#9467bd',  # Purple (distinguishable)
    '#1f77b4',  # Blue (safe for all colorblind types) - swapped from position 0
    '#8c564b',  # Brown (earth tone)
    '#e377c2'   # Pink (distinct from other colors)
]
colors_verdictdb = ['#7f7f7f', '#bcbd22']  # Gray and olive (colorblind-safe)
color_inference = '#17becf'  # Cyan (neutral and distinguishable)

X = ["S1.1", "S1.2", "S1.3", "S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]

# e2e = { # actual numbers
#     'Compilation': np.array([22.8,22.8,23,24.8,24,23.6,24.4,43.2,47.2,45.6]),
#     'Inference': np.array([64.3198384000016,26.783301200021,21.3875056000234,46.1069563999445,142.064021999977,31.6860601999906,29.7971963999771,51.7050286000085,104.852731400024,385.365180400095]),
#     'Python Overhead': np.array([17.2801615999984,8.416698799979,6.0124943999766,10.8930436000555,7.135978000023,6.9139398000094,30.8028036000229,7.2949713999915,9.147268599976,8.834819599905]),
# }

e2e = { # percentage
    'parse': np.array([60.4,
        60.2,
        61,
        77.8,
        74,
        79.2,
        83.6,
        84.2,
        84.8,
        84.4,
        88.8,
        91,
        74.0]),
    'TensorLowering': np.array([26.8,
        26.6,
        26,
        36.6,
        35.8,
        32.8,
        33.8,
        34.4,
        32.8,
        33,
        40.4,
        41.4,
        44.6]),
    'TensorToSetar': np.array([8,
        9.4,
        9.8,
        9,
        9.8,
        10.2,
        10.6,
        10.8,
        11,
        13.2,
        13.2,
        13.2,
        14.6]),
    'O1': np.array([13.4,
        13.2,
        13.6,
        19.2,
        20,
        22.2,
        24.6,
        23.6,
        23.4,
        23.8,
        28,
        29.6,
        30.2]),
    'O2': np.array([3.6,
        4,
        4,
        6.6,
        5.4,
        4.4,
        5,
        5,
        5,
        5,
        6.8,
        6.4,
        5.8]),
    'SetarToTensor': np.array([3.4,
        3.8,
        4,
        3,
        3,
        3.4,
        3,
        3,
        3,
        4,
        3.2,
        3.4,
        3.4]),
    'Codegen': np.array([25.2,
        25.8,
        26.4,
        25.6,
        29,
        25.8,
        25.2,
        24.8,
        24.4,
        24.8,
        29.4,
        30.2,
        33.8]),
    'Python Overhead': np.array([13.667833389155566,
        13.237541563622653,
        13.194983218610286,
        14.742141606286168,
        13.966308234259486,
        14.0672836413607,
        13.762242025323213,
        14.173333216644824,
        13.86125001590699,
        14.093991583772004,
        14.05025820620358,
        13.820308417081833,
        14.257883416675032]),
    'Inference': np.array([2.653,
     2.533, 2.738, 16.133, 6.697,
      4.899, 10.223, 32.694, 9.837,
       9.757, 10.154, 18.371, 92.330])
}



verdictdb = {
    'Preprocessing (VerdictDB)' : np.array([11.250,
    6.683,
    5.754,
    53.185,
    23.121,
    10.063,
    43.967,
    111.028,
    14.248,
    0,
    18.2080,
    36.702,
    50.096]),
    'DB execution (VerdictDB)' : np.array([48.6,40.8,
    46.8,426.6,488.4,247.8,528.6,316.4,204.4,0,718.2,535.4,387.4])
}

# numpy and pytorch have similar performance
cmp_ovh = list(zip(*e2e.values()))
cmp_ovh_q = dict(zip(X, cmp_ovh))

# d = dict(sorted(cmp_ovh_q.items(), key=lambda x: sum(x[1]))) # sort based on total time of overhead (compilation + python overhead)
d = cmp_ovh_q

X2 = list(d.keys())
X_axis = np.arange(len(X2)) 
e2e = list(zip(*d.values()))
# Add inference data - match it to the actual queries being plotted
inference_dict = dict(zip(X, e2e[-1]))
# e2e = {
#     'TensorLowering (BayesAQP)': np.array(e2e[1]),
#     'TensorToSetar (BayesAQP)': np.array(e2e[2]),
#     'O1 (BayesAQP)': np.array(e2e[3]),
#     'O2 (BayesAQP)': np.array(e2e[4]),
#     'SetarToTensor (BayesAQP)': np.array(e2e[5]),
#     'Codegen (BayesAQP)': np.array(e2e[6]),
#     'Python Overhead (BayesAQP)': np.array(e2e[7])
# }
e2e = {
    'IR Translation (BayesAQP)': np.array(e2e[1]) + np.array(e2e[2]) + np.array(e2e[5]),
    'O1 (BayesAQP)': np.array(e2e[3]),
    'O2 (BayesAQP)': np.array(e2e[4]),
    'Codegen (BayesAQP)': np.array(e2e[6]),
    'Python Overhead (BayesAQP)': np.array(e2e[7])
}


# Sort inference data to match X2 order
inference_sorted = {key: inference_dict[key] for key in X2}
inference = {
    'Inference (BayesAQP)': np.array(list(inference_sorted.values()))
}

verdictdb_ovh = list(zip(*verdictdb.values()))
verdictdb_ovh_q = dict(zip(X, verdictdb_ovh))

verdictdb_sort = {key: verdictdb_ovh_q[key] for key in X2}
verdictdb_values = list(zip(*verdictdb_sort.values()))
verdictdb = {
    'Preprocessing (VerdictDB)' : np.array(verdictdb_values[0]),
    'DB execution (VerdictDB)' : np.array(verdictdb_values[1])
}


base_width = .7  # Base width for 4 queries (S3)
# Calculate proportional width so bars appear similar across subplots

# Create single plot with all queries (compressed height)
fig, ax = plt.subplots(figsize=(20, 8))
fig.patch.set_facecolor('white')
# Add subtle shadow effect
fig.patch.set_edgecolor('#bdc3c7')
fig.patch.set_linewidth(2)

# Use all queries in X2 order
X_axis = np.arange(len(X2))
bottom = np.zeros(len(X2))
compilation_bottom = np.zeros(len(X2))

# Plot BayesAQP components with enhanced styling
count = 0
for phase, e2e_data in e2e.items():
    color = colors_bayesaqp[count % len(colors_bayesaqp)]
    p = ax.bar(X_axis-base_width/4, e2e_data, base_width/2, label=phase, 
               bottom=bottom, color=color, edgecolor='white', linewidth=1.0, alpha=0.85)
    bottom += e2e_data
    compilation_bottom += e2e_data
    
    if phase != 'Environment Pre-load':
        ax.bar_label(p, fmt='%.0f', label_type='center', fontsize=18, # weight='bold' 
                    color='black')
    count += 1

# Add inference layer
for phase, inf_data in inference.items():
    color = color_inference
    p_inf = ax.bar(X_axis-base_width/4, inf_data, base_width/2, 
                   label=phase, bottom=bottom,
                   color=color, edgecolor='white', linewidth=1.0, alpha=0.85)
    ax.bar_label(p_inf, fmt='%.1f', label_type='center', fontsize=18, color='black') # , weight='bold'

# Add VerdictDB components
bottom_vdb = np.zeros(len(X2))
for phase, vdb in verdictdb.items():
    if 'Preprocessing' in phase:
        color = colors_verdictdb[0]
    else:
        color = '#FF9999'
    
    p2 = ax.bar(X_axis+base_width/4, vdb, base_width/2, 
                label=phase, bottom=bottom_vdb,
                color=color, edgecolor='white', linewidth=1.0, alpha=0.85)
    bottom_vdb += vdb
    ax.bar_label(p2, fmt=lambda x: f'{x:.0f}' if x > 0 else '', label_type='center', 
                 fontsize=18, color='black') # , weight='bold'

# Add "No result" annotation for VerdictDB where values are zero
for i, query in enumerate(X2):
    if query == 'S3.4':  # S3.4 has zero values for VerdictDB
        total_vdb = sum(vdb[i] for vdb in verdictdb.values())
        if total_vdb == 0:
            ax.text(X_axis[i]+base_width/4, 10, 'No result', 
                   fontsize=22, color='red',
                   ha='center', va='bottom', rotation=90) # weight='bold'

# Add mean compilation line with label
mean_compile = np.mean(compilation_bottom)
ax.axhline(mean_compile, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(-0.47, mean_compile + 3, f'Mean Overhead (BayesAQP) ≈ {mean_compile:.0f}ms', 
        fontsize=18, color='red', ha='left', va='bottom') # , weight='bold'

# Customize plot with professional styling
# Remove log scale and set y-axis limit to 200ms
ax.set_ylim(0, 160)

# Enhanced grid styling
ax.grid(axis='y', linestyle='-', alpha=0.3, color='#bdc3c7', linewidth=0.8)
ax.set_axisbelow(True)

# Improve plot borders and background
ax.set_facecolor('#fafafa')
for spine in ax.spines.values():
    spine.set_color('#2c3e50')
    spine.set_linewidth(1.2)

ax.set_xticks(X_axis)
ax.set_xticklabels(X2, fontsize=FONTSIZE+1)

# Set labels without bold formatting
ax.set_xlabel("Queries", size=FONTSIZE+1)
ax.set_ylabel("End-to-End Time (ms)", size=FONTSIZE)

# Improve tick label sizes
ax.tick_params(axis='y', labelsize=FONTSIZE)

# Add an elegant legend
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.08, 0.86, 0.9, 0.08), 
           ncol=4, frameon=True, fancybox=True, shadow=False, facecolor='white',
           prop={'size': FONTSIZE, 'weight': 'normal'}, columnspacing=2.0, 
           handletextpad=0.8, handlelength=2.0, mode="expand", borderaxespad=0)

# Style the legend frame
legend.get_frame().set_facecolor('#f8f9fa')
legend.get_frame().set_edgecolor('#2c3e50')
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_alpha(0.95)

# Main title removed for cleaner look
# fig.suptitle('Compilation Overhead + Inference: BayesAQP vs VerdictDB by Query Pattern', 
#              fontsize=FONTSIZE+2, fontweight='bold', y=0.98)
ax.set_xlim(-0.5, len(X2) - 0.5)
# Optimize layout for compressed single plot
plt.tight_layout()
plt.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.98)

# Save with high DPI for better quality
fig.savefig("progs/plots/ssb/overhead/sf50/ssb50gb_overhead.pdf", 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

# Optionally show the plot
# plt.show()

# ax.set_xlim(-0.5, len(X2) - 0.5)
# plt.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.95)
