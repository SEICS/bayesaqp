import math
import os
from decimal import Decimal 
import re
import pickle
import numpy as np
import pandas as pd

# Needs to run postgres to have .csv results files
def read_csv_to_dict(dir):
    res = dict()
    for filename in sorted(os.listdir(dir)):
        if filename.startswith("psql_q") and filename.endswith(".csv"): 
            # print("".join(re.findall(r'\d+', filename)))
            query_no = int("".join(re.findall(r'\d+', filename)))
            f = open(os.path.join(dir, filename), "r").readlines()
            f = f[1:] # drop header
            new_f = []
            for row in f:
                new_row = []
                row = row.split(",")
                for attr in row:
                    try:
                        attr = float(attr) 
                    except:
                        pass
                    new_row.append(attr)
                new_f.append(tuple(new_row))
            res[query_no] = new_f
        else:
            continue
    return res

def compute_mean_selectivity(target_dir, ground_truth_dir=None, use_pandas=True):
    target = read_csv_to_dict(target_dir)
    sel_dict = dict()

    if ground_truth_dir is None:
        raise ValueError("ground_truth_dir is required since SSB has a star schema")
    else:
        ground_truth = read_csv_to_dict(ground_truth_dir)
        assert len(target) == len(ground_truth)
        for query_no in target:
            tar = target[query_no]
            if len(tar[0])>0: # if the inner tuple is not empty. Because len([tuple()]) = 1.
                gt = ground_truth[query_no]
                if use_pandas:
                    # Use ultra-fast pandas version for large datasets
                    sel_dict[query_no] = np.mean(compute_group_selectivity_pandas(gt, tar)) * 100
                else:
                    # Use optimized dictionary-based version
                    sel_dict[query_no] = np.mean(compute_group_selectivity(gt, tar)) * 100
            else:
                continue

    return dict(sorted(sel_dict.items()))

def compute_group_selectivity(true, pred):
    sel = []
    if isinstance(true, list) or isinstance(true, np.ndarray):
        # Optimize: Use dictionary for O(1) lookup instead of O(n²) nested loops
        true_dict = {}
        
        # Build dictionary from ground truth data
        for t in true:
            if len(t) > 0:
                # Create key from all columns except the last (count) column
                key_parts = []
                for val in t[:-1]:
                    if isinstance(val, str):
                        if val.replace(".", "").isdigit():
                            key_parts.append(float(val))
                        else:
                            key_parts.append(val)
                    else:
                        key_parts.append(val)
                key = tuple(key_parts)
                true_dict[key] = float(t[-1])
        
        # Look up predictions in dictionary
        for p in pred:
            if len(p) > 0 and not math.isnan(float(p[-1])):
                # Create key from prediction
                key_parts = []
                for val in p[:-1]:
                    if isinstance(val, str):
                        if val.replace(".", "").isdigit():
                            key_parts.append(float(val))
                        else:
                            key_parts.append(val)
                    else:
                        key_parts.append(val)
                key = tuple(key_parts)
                
                # O(1) lookup instead of O(n) search
                if key in true_dict:
                    true_count = true_dict[key]
                    sel.append(float(p[-1]) / true_count)
                    
    elif isinstance(true, float) or isinstance(true, int):
        for p in pred:
            if not math.isnan(float(p[-1])):
                sel.append(float(p[-1])/true)
    return sel

def compute_group_selectivity_pandas(true, pred):
    """
    Ultra-fast pandas-based version for large datasets.
    Converts data to DataFrames and uses vectorized operations.
    """
    if isinstance(true, (list, np.ndarray)) and isinstance(pred, (list, np.ndarray)):
        if len(true) == 0 or len(pred) == 0:
            return []
            
        # Convert to DataFrames
        true_df = pd.DataFrame(true)
        pred_df = pd.DataFrame(pred)
        
        if len(true_df.columns) == 0 or len(pred_df.columns) == 0:
            return []
        
        # Last column is count, others are keys
        count_col = len(true_df.columns) - 1
        key_cols = list(range(count_col))
        
        # Handle case where there are no key columns (only count column)
        if len(key_cols) == 0:
            # Only one column (count), treat as simple case
            true_counts = pd.to_numeric(true_df[count_col], errors='coerce').dropna()
            pred_counts = pd.to_numeric(pred_df[count_col], errors='coerce').dropna()
            if len(true_counts) > 0 and len(pred_counts) > 0:
                # For single values, compute selectivity directly
                return [pred_counts.iloc[0] / true_counts.iloc[0]]
            else:
                return []
        
        # Convert string numbers to float in key columns for consistent joining
        for col in key_cols:
            true_df[col] = true_df[col].apply(lambda x: float(x) if isinstance(x, str) and x.replace(".", "").isdigit() else x)
            pred_df[col] = pred_df[col].apply(lambda x: float(x) if isinstance(x, str) and x.replace(".", "").isdigit() else x)
        
        # Ensure count columns are numeric
        true_df[count_col] = pd.to_numeric(true_df[count_col], errors='coerce')
        pred_df[count_col] = pd.to_numeric(pred_df[count_col], errors='coerce')
        
        # Remove rows with NaN counts
        true_df = true_df.dropna(subset=[count_col])
        pred_df = pred_df.dropna(subset=[count_col])
        
        # Merge on key columns to match rows
        merged = pred_df.merge(true_df, on=key_cols, suffixes=('_pred', '_true'), how='inner')
        
        # Calculate selectivity (pred_count / true_count)
        if len(merged) > 0:
            selectivity = merged[f'{count_col}_pred'] / merged[f'{count_col}_true']
            return selectivity.tolist()
        else:
            return []
    elif isinstance(true, (float, int)):
        # Simple case: single true value
        sel = []
        for p in pred:
            if not math.isnan(float(p[-1])):
                sel.append(float(p[-1]) / true)
        return sel
    else:
        return []

if __name__ == "__main__":
    import time
    
    ce_no_sel_dir = "progs/plots/ssb/selectivity/sf50/postgres/cardEst/no_sel" 
    ce_sel_dir = "progs/plots/ssb/selectivity/sf50/postgres/cardEst/sel" 
    gb_no_sel_dir = "progs/plots/ssb/selectivity/sf50/postgres/groupby/no_sel"
    gb_sel_dir = "progs/plots/ssb/selectivity/sf50/postgres/groupby/sel"
    
    print("Computing cardEst selectivity...")
    start_time = time.time()
    sel = compute_mean_selectivity(ce_sel_dir, ce_no_sel_dir, use_pandas=True)
    cardest_time = time.time() - start_time
    print(f"cardEst selectivity (took {cardest_time:.2f}s): {sel}")
    
    # print("\nComputing groupby selectivity with optimized pandas version...")
    # start_time = time.time()
    # no_sel = compute_mean_selectivity(gb_sel_dir, gb_no_sel_dir, use_pandas=True)
    # groupby_time_pandas = time.time() - start_time
    # print(f"groupby selectivity with pandas (took {groupby_time_pandas:.2f}s): {no_sel}")
    
    print("\nComputing groupby selectivity with optimized dictionary version...")
    start_time = time.time()
    no_sel_dict = compute_mean_selectivity(gb_sel_dir, gb_no_sel_dir, use_pandas=False)
    groupby_time_dict = time.time() - start_time
    print(f"groupby selectivity with dict (took {groupby_time_dict:.2f}s): {no_sel_dict}")
    
    # print(f"\nPerformance comparison:")
    # print(f"  Dict vs Pandas speedup: {groupby_time_pandas / groupby_time_dict:.1f}x faster")

    # Performance comparison:
    # Dict vs Pandas speedup: 1.1x faster