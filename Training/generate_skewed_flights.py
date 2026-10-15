import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_bounded_zipfian(n, alpha, size):
    """
    Generates 'size' samples from a bounded Zipfian distribution on [1, n].
    P(k) \propto 1/k^alpha
    """
    if alpha == 0:
        return np.random.randint(1, n + 1, size)
    
    try:
        from scipy.stats import zipfian
        return zipfian.rvs(alpha, n, size=size)
    except ImportError:
        k = np.arange(1, n + 1)
        weights = 1.0 / (k ** alpha)
        probs = weights / np.sum(weights)
        cum_probs = np.cumsum(probs)
        u = np.random.random(size)
        samples = np.searchsorted(cum_probs, u) + 1
        return samples

def get_unique_values_and_map(df, key_col, value_col=None):
    """
    Returns unique keys and a mapping from key -> value (if value_col provided)
    """
    if value_col:
        # distinct pairs
        distinct = df[[key_col, value_col]].drop_duplicates(subset=[key_col])
        keys = distinct[key_col].values
        mapping = dict(zip(distinct[key_col], distinct[value_col]))
        return keys, mapping
    else:
        keys = df[key_col].unique()
        return keys, None

def generate_skewed_flights(src_path, dst_base_path, skew_params=[0.0, 0.5, 1.0, 1.5, 2.0], mode="join_key"):
    """
    Generates skewed variants of the Flights dataset.
    
    Note on 'mode':
    - Flights is a single denormalized table, so strictly speaking there are no joins.
    - 'join_key' mode: Skews 'origin', 'dest', and 'unique_carrier'. 
      These act as 'implicit foreign keys' to Airport/Carrier dimensions in a star schema view.
    - 'group_by' mode: Skews 'year_date'. 
      This represents a temporal dimension often used for grouping/filtering.
    """
    print(f"Generating skewed Flights datasets.")
    print(f"Source: {src_path}")
    print(f"Destination Base: {dst_base_path}")
    print(f"Mode: {mode}")

    # Flights is usually a single file
    # We expect 'dataset_sampled.csv' or 'flights.csv'
    # schema.py says 'dataset_sampled'
    
    filename = 'dataset_sampled.csv'
    src_file = os.path.join(src_path, filename)
    if not os.path.exists(src_file):
        # Fallback
        filename = 'flights.csv'
        src_file = os.path.join(src_path, filename)
        if not os.path.exists(src_file):
             print(f"Error: Source file not found in {src_path}")
             return

    # Scan columns
    cols_to_scan = ['origin', 'origin_state_abr', 'dest', 'dest_state_abr', 'unique_carrier']
    if mode == "group_by":
        cols_to_scan.append('year_date')

    # Read unique values by iterating chunks
    unique_origins = set()
    unique_dests = set()
    unique_carriers = set()
    unique_dates = set()
    
    # We need mappings origin->state, dest->state. 
    # Since FD holds, any occurrence gives the correct mapping.
    origin_map = {}
    dest_map = {}
    
    chunksize = 1000000
    try:
        reader = pd.read_csv(src_file, usecols=lambda x: x in cols_to_scan, chunksize=chunksize)
        for chunk in tqdm(reader, desc="Scanning domains"):
            # Origin
            if 'origin' in chunk.columns:
                if 'origin_state_abr' in chunk.columns:
                    # Drop dups in chunk
                    tmp = chunk[['origin', 'origin_state_abr']].drop_duplicates('origin')
                    for _, row in tmp.iterrows():
                        origin_map[row['origin']] = row['origin_state_abr']
                unique_origins.update(chunk['origin'].dropna().unique())
            
            # Dest
            if 'dest' in chunk.columns:
                if 'dest_state_abr' in chunk.columns:
                    tmp = chunk[['dest', 'dest_state_abr']].drop_duplicates('dest')
                    for _, row in tmp.iterrows():
                        dest_map[row['dest']] = row['dest_state_abr']
                unique_dests.update(chunk['dest'].dropna().unique())
                
            # Carrier
            if 'unique_carrier' in chunk.columns:
                unique_carriers.update(chunk['unique_carrier'].dropna().unique())
            
            # Date
            if 'year_date' in chunk.columns and mode == "group_by":
                unique_dates.update(chunk['year_date'].dropna().unique())
                
    except Exception as e:
        print(f"Error scanning domains: {e}")
        return

    origins_list = sorted(list(unique_origins))
    dests_list = sorted(list(unique_dests))
    carriers_list = sorted(list(unique_carriers))
    dates_list = sorted(list(unique_dates))
    
    print(f"Found {len(origins_list)} origins, {len(dests_list)} dests, {len(carriers_list)} carriers, {len(dates_list)} dates.")

    for skew in skew_params:
        print(f"\n--- Generating data for skew={skew} ---")
        dst_dir = os.path.join(dst_base_path, f"{mode}_skew_{skew}")
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, filename)
        
        if os.path.exists(dst_file):
            print(f"{filename} already exists, skipping.")
            continue
            
        print(f"Processing {filename}...")
        first = True
        
        try:
            reader = pd.read_csv(src_file, chunksize=chunksize)
            
            for chunk in tqdm(reader):
                n_rows = len(chunk)
                
                if mode == "join_key":
                    # Skew "Pseudo-Join Keys" (Dimensions): Origin, Dest, Carrier
                    
                    # Origin
                    if 'origin' in chunk.columns:
                        indices = generate_bounded_zipfian(len(origins_list), skew, n_rows) - 1
                        new_origins = np.array(origins_list)[indices]
                        chunk['origin'] = new_origins
                        # Update state
                        if 'origin_state_abr' in chunk.columns:
                            # Map efficiently
                            chunk['origin_state_abr'] = [origin_map.get(o, None) for o in new_origins]
                            
                    # Dest
                    if 'dest' in chunk.columns:
                        indices = generate_bounded_zipfian(len(dests_list), skew, n_rows) - 1
                        new_dests = np.array(dests_list)[indices]
                        chunk['dest'] = new_dests
                        # Update state
                        if 'dest_state_abr' in chunk.columns:
                            chunk['dest_state_abr'] = [dest_map.get(d, None) for d in new_dests]
                            
                    # Carrier
                    if 'unique_carrier' in chunk.columns:
                        indices = generate_bounded_zipfian(len(carriers_list), skew, n_rows) - 1
                        chunk['unique_carrier'] = np.array(carriers_list)[indices]
                        
                elif mode == "group_by":
                    # Skew 'year_date'
                    if 'year_date' in chunk.columns:
                        indices = generate_bounded_zipfian(len(dates_list), skew, n_rows) - 1
                        chunk['year_date'] = np.array(dates_list)[indices]
                
                mode_write = 'w' if first else 'a'
                header = first
                chunk.to_csv(dst_file, index=False, header=header, mode=mode_write)
                first = False
                
        except Exception as e:
            print(f"Failed to process chunk: {e}")

    print("\nData generation complete.")

# Note: For 'group_by' mode in Flights, if we treat Origin/Dest as Keys, 
# then `year_date` is the primary non-key grouper.
# I will update the scanning logic to include year_date if mode is group_by.
