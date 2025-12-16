import numbers
import numpy as np
import pandas as pd
from collections import OrderedDict
from time import perf_counter
import hashlib
import random
import heapq

def categorical_qcut(series, q, start_value):
    """Perform equi-depth histogram construction over categorical data."""
    bin_freq = 1 / q
    value_counts = series.value_counts(normalize=True) 
    bins = {}
    n_distinct = {} # Count the number of distinct values per bin
    valname_to_bin = {}
    bin_to_valname = {}
    values = value_counts.index.values # domain values
    freqs = value_counts.values # normalized frequency 
    curr = start_value # current bin
    cum_freq = 0
    bin_start_idx = 0
    
    for i, (val, freq) in enumerate(zip(values, freqs)):
        cum_freq += freq
        valname_to_bin[val] = curr
        bins[val] = curr
        
        if cum_freq >= bin_freq or (i + 1) == len(values):
            # Get all values in this bin
            bin_values = values[bin_start_idx:i+1]
            bin_freqs = freqs[bin_start_idx:i+1]
            
            # Sort values for this bin
            sorted_indices = np.argsort(bin_values)
            sorted_bin_values = bin_values[sorted_indices].tolist()
            sorted_bin_freqs = bin_freqs[sorted_indices]
            
            bin_to_valname[curr] = sorted_bin_values
            
            # Create n_distinct mapping with normalized frequencies within bin
            n_distinct[curr] = {
                val: sorted_bin_freqs[j] / cum_freq 
                for j, val in enumerate(sorted_bin_values)
            }
            
            curr += 1
            cum_freq = 0
            bin_start_idx = i + 1
    
    return n_distinct, valname_to_bin, bin_to_valname


def discretize_series(series: pd.Series, n_mcv, n_bins, is_continuous=False, 
                     continuous_bins=None, drop_na=True, enable_timing=False):
    """
    Perform equi-depth histogram constructions over data, where the number of bins for continuous 
    attribute is double the number of bins for categorical attribute. 
    
    Args:
        enable_timing: Set to True to enable performance timing (adds overhead)

    Returns:
        bin_label: the bin labels for the discretized series
        n_distinct: the number of distinct values per bin 
        mapping: the mapping from the bin labels to the intervals
        bins_intervals: the mean values of the bins
        domains: the domains of the discretized series
        valname_to_bin: the mapping from the original domain values to the bin labels
        bin_to_valname: the mapping from the bin labels to the original domain values
    """
    
    def time_it(func, *args, **kwargs):
        if enable_timing:
            start = perf_counter()
            res = func(*args, **kwargs)
            print(f"{func.__name__} took: {(perf_counter() - start) * 1000:.3f}ms")
            return res
        else:
            return func(*args, **kwargs)
    
    s = series.copy()
    n_distinct = {} # Count the number of distinct values per bin
    valname_to_bin = {}
    bin_to_valname = {}

    if is_continuous:
        mapping = {}
        if not continuous_bins:
            continuous_bins = n_bins * 2
        domains = (s.min(), s.max())
        bin_label, bin_edges = time_it(pd.qcut(s, q=continuous_bins, duplicates='drop', labels=False, retbins=True)) # return bin labels and the edges to decided nth quantile
        intervals = pd.cut(s, bins=bin_edges, include_lowest=True) # the data series after cutting, exactly the same as qcut but faster
        
        if drop_na:
            bna = int(bin_label.nunique() + 1)
            bin_label = bin_label.cat.add_categories(bna)
            bin_label = bin_label.fillna(bna)

        def create_mappings():
            bins_intervals = s.groupby(bin_label).mean().values # np.ndarray type
            mapping = dict(zip(bin_label, intervals)) # nth bucket -> interval. e.g., {0 -> (-2.554, -1.325]}
            
            # bin_to_valname and valname_to_bin creation
            grouped = s.groupby(bin_label)
            bin_to_valname = {bin_id: group.unique().tolist() for bin_id, group in grouped}
            
            # Vectorized valname_to_bin creation
            valname_to_bin = {}
            for bin_id, values in bin_to_valname.items():
                for val in values:
                    valname_to_bin[val] = bin_id
                    
            return bins_intervals, mapping # No need to store bin_to_valname and valname_to_bin for continous data
        
        bins_intervals, mapping = time_it(create_mappings)
        
        return bin_label, None, OrderedDict(sorted(mapping.items())), bins_intervals, domains, None, None

    if s.dtype == 'object': # Remove trailing whitespace
        s = s.str.strip()
    
    domains = s.unique().tolist()
    value_counts = s.value_counts()
    if not domains and not value_counts.empty:
        domains = value_counts.index.tolist()
    elif not domains: 
        domains = []
    
    n_mcv = len(value_counts) if n_mcv == -1 else n_mcv
    n_largest = value_counts.nlargest(n_mcv) # Most common values
    most_common_vals = set(n_largest.index)
    most_common_mask = s.isin(most_common_vals)
    
    # Encoding most common categories to int categories (i.e., Each falls into one bucket with no other values)
    val = 0
    for i in n_largest.index:
        bin_to_valname[val] = [i]
        valname_to_bin[i] = val # i is data value, val is the #Bin
        n_distinct[val] = {i: 1} # unbinned unique value has a weight of 1 always
        val += 1
    
    n_least_common = s[~most_common_mask].nunique()
    n_bins = min(n_least_common, n_bins)
    
    if n_least_common > 0:
        nl_distinct, nl_valname_to_bin, nl_bin_to_valname = categorical_qcut(
            s[~most_common_mask], n_bins, val
        )
        valname_to_bin.update(nl_valname_to_bin)
        bin_to_valname.update(nl_bin_to_valname)
        n_distinct.update(nl_distinct)
    
    temp = s.map(valname_to_bin) # Map the original value to encoded value but will introduce NaNs if no matching keys
    
    if drop_na:
        temp = temp.fillna(n_mcv + n_bins + 1) # Replace np.nan with some integer that is not in valname_to_bin
    
    return temp, n_distinct, None, None, domains, valname_to_bin, bin_to_valname


def scale_cat_attribute_domain_size(series: pd.Series, n_bins, is_continuous, drop_na=True):
    """
    Vary the domain size of categorical data using hash-based mapping.
    Guarantees exactly n_bins unique synthetic values.
    
    Args:
        series: Input pandas Series
        n_bins: Target domain size (number of unique synthetic values)
        is_continuous: Must be False for categorical data
        drop_na: Whether to handle NaN values

    Returns:
        temp: Series with synthetic values
    """
    def hash_bin(v):
        return int(hashlib.md5(str(v).encode()).hexdigest(), 16) % n_bins

    if is_continuous:
        raise NotImplementedError("Continuous data is not supported for now")
    
    s = series.copy()
    if s.dtype == 'object': # Remove trailing whitespace
        s = s.str.strip()

    domain = sorted(s.dropna().unique(), key=str)
    if len(domain) < n_bins:
        raise ValueError(f"Not enough domain size ({len(domain)}) to create {n_bins} bins.")

    vc = s.value_counts()
    mapping = {} # domain value -> "synthetic_value_{b}"
    bin_vals = [[] for _ in range(n_bins)]
    bin_mass = [0] * n_bins
    for v in domain:
        b = hash_bin(v)
        mapping[v] = f"synthetic_value_{b}"
        bin_vals[b].append(v)
        bin_mass[b] += int(vc.get(v, 0))
    # Identify empty bins
    missing = [b for b in range(n_bins) if not bin_vals[b]]
    if missing:
        # Max-heap of donors by current mass; bins must have >1 values to donate
        heap = [(-bin_mass[b], b) for b in range(n_bins) if len(bin_vals[b]) > 1]
        heapq.heapify(heap)

        while missing:
            if not heap:
                raise RuntimeError("Not enough donor capacity to fill all missing bins.")
            _, donor = heapq.heappop(heap)
            if len(bin_vals[donor]) <= 1:
                continue  # re-check next donor

            # peel least-frequent (tie-break lexicographically)
            v_peel = min(bin_vals[donor], key=lambda v: (int(vc.get(v,0)), str(v)))
            bin_vals[donor].remove(v_peel)
            bin_mass[donor] -= int(vc.get(v_peel, 0))

            mb = missing.pop()  # take one missing bin
            bin_vals[mb].append(v_peel)
            bin_mass[mb] += int(vc.get(v_peel, 0))
            mapping[v_peel] = f"synthetic_value_{mb}"

            # donor may still have >1 values; push back
            if len(bin_vals[donor]) > 1:
                heapq.heappush(heap, (-bin_mass[donor], donor))

    temp = s.map(lambda v: mapping.get(v) if pd.notna(v) else None)
    
    if drop_na and temp.isna().any():
        temp = temp.fillna(f"synthetic_value_{n_bins}")  # Use an extra bucket for NaN

    # Final sanity check
    used = set(temp.dropna().unique())
    assert len(used) == n_bins, f"Expected {n_bins} bins, got {len(used)}"
    
    return temp