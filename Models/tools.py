import pandas as pd
from collections import OrderedDict

def categorical_qcut(series, q, start_value): 
    """Computes categorical quantiles of a pandas.Series objects."""
    bin_freq = 1 / q
    value_counts = series.value_counts(normalize=True).sort_index()
    bins = {}
    n_distinct = {} # Count the number of distinct values per bin
    valname_to_bin = {}
    bin_to_valname = {}
    values_in_bin = []
    freq_in_bin = []
    cum_freq = 0
    
    value = start_value # start No.Bin for binning least common values
    for i, (val, freq) in enumerate(value_counts.items()):
        values_in_bin.append(val)
        freq_in_bin.append(freq)
        cum_freq += freq
        valname_to_bin[val] = value
        
        if cum_freq >= bin_freq or (i+1) == len(value_counts):
            values_in_bin = sorted(values_in_bin)
            n_distinct[value] = dict()
            for j, v in enumerate(values_in_bin):
                bins[v] = value
                n_distinct[value][v] = freq_in_bin[j]/cum_freq
                
            bin_to_valname[value] = values_in_bin
            values_in_bin = [] 
            freq_in_bin = [] 
            cum_freq = 0 
            value += 1 # Increment the No.Bin
        
    return n_distinct, valname_to_bin, bin_to_valname

def discretize_series(series: pd.Series, n_mcv, n_bins, is_continuous=False, continuous_bins=None, drop_na=True):
    """
    Map every value to category, binning the small categories if there are more than n_mcv categories.
    Map intervals to categories for efficient model learning
    return:
    s: discretized series
    n_distinct: number of distinct values in a mapped category (could be empty)
    valname_to_bin: encode the original value to new category 
    mapping: map the new category to pd.Interval (for continuous attribute only)
    """

    s = series.copy()
    n_distinct = dict()
    valname_to_bin = dict() # data value maps to its corresponding bin. Key: data value, Value: #Bin
    bin_to_valname = dict() # Reversed of valname_to_bin. Key: #Bin, Value: list of data values in this bin.
    if is_continuous: 
        mapping = dict()
        if not continuous_bins:
            continuous_bins = n_bins * 2

        domains = (s.min(), s.max())
        intervals = pd.qcut(s, q=continuous_bins, duplicates='drop') # retbins not necessarily return bin's index from 0
        ds = pd.qcut(s, q=continuous_bins, duplicates='drop', labels=False) # Discretized series
        if drop_na:
            bna = int(ds.nunique()+1)
            ds = ds.cat.add_categories(bna)
            ds = ds.fillna(bna)  # Replace np.nan with some integer that is not in valname_to_bin

        new_name = f"binned_{ds.name}"
        ds = ds.rename(new_name)
        ds2 = pd.concat([s,ds],axis=1)
        dsg = ds2.groupby(by=[new_name])
        bins_intervals = dsg.mean().to_numpy().flatten()
        ds3 = pd.concat([ds,intervals],axis=1)
        mapping = dict(zip(ds3[ds.name],ds3[intervals.name]))
        bin_to_valname = dsg[s.name].unique().to_dict()
        valname_to_bin = dict(ds2.drop_duplicates().to_dict('split')['data'])
        return ds, None, OrderedDict(sorted(mapping.items())), bins_intervals, domains, valname_to_bin, bin_to_valname 
        
    # Remove trailing whitespace
    if s.dtype == 'object':
        s = s.str.strip()
    domains = list(s.unique())
    # Treat most common values
    value_counts = s.value_counts()
    if not domains:
        if not value_counts.empty:
            domains = list(value_counts.index)
        else:
            domains = []

    n_mcv = len(value_counts) if n_mcv == -1 else n_mcv
    n_largest = value_counts.nlargest(n_mcv)
    most_common_vals = set(n_largest.index)
    most_common_mask = s.isin(most_common_vals)
    
    #encoding most common categories to int categories
    val = 0
    n_distinct = dict()
    for i in n_largest.index:
        bin_to_valname[val] = [i] # Don't bin the most common values
        valname_to_bin[i] = val # i is data value, val is the No. of Bin
        n_distinct[val] = {i:1} # unbinned unique value has a weight of 1 always
        val += 1
        
    # Treat least common values
    n_least_common = s[~most_common_mask].nunique()
    n_bins = min(n_least_common, n_bins) # bin unique values values if too many of them
    if n_least_common > 0:
        #encoding the least common string category to int category
        nl_distinct, nl_valname_to_bin, nl_bin_to_valname = categorical_qcut(s[~most_common_mask], n_bins, val)
        valname_to_bin.update(nl_valname_to_bin)
        bin_to_valname.update(nl_bin_to_valname)
        n_distinct.update(nl_distinct)
    
    #map the original value to encoded value
    temp = s.copy() # series.copy()
    temp = temp.map(valname_to_bin) # will introduce NaNs if no matching keys
    del s
    
    if drop_na:
        temp = temp.fillna(n_mcv+n_bins+1, inplace=True) # Replace np.nan with some integer that is not in valname_to_bin

    return temp, n_distinct, None, None, domains, valname_to_bin, bin_to_valname #, sorted(range(len(bin_to_valname))) #self.valname_to_bin[col], self.bin_to_valname[col]