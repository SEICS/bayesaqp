import os
import shutil
from Training.utils import get_ssb_colname, usefulCols_ssb
import numpy as np
import pandas as pd
from tqdm import tqdm
from Schemas.ssb.schema import gen_5gb_ssb_schema

def generate_bounded_zipfian(n, alpha, size):
    """
    Generates 'size' samples from a bounded Zipfian distribution on [1, n].
    P(k) is proportional to 1/k^alpha
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
        
        # Use inverse transform sampling with searchsorted on CDF
        cum_probs = np.cumsum(probs)
        u = np.random.random(size)
        
        # searchsorted finds indices where elements should be inserted to maintain order
        # indices will be 0 to n-1. We want 1 to n.
        samples = np.searchsorted(cum_probs, u) + 1
        return samples

def generate_skewed_ssb(src_path, dst_base_path, skew_params=[0.0, 0.5, 1.0, 1.5, 2.0]):
    print(f"Generating skewed SSB datasets.")
    print(f"Source: {src_path}")
    print(f"Destination Base: {dst_base_path}")
    
    # Get schema info for table sizes
    try:
        schema = gen_5gb_ssb_schema(os.path.join(src_path, '{}.csv'))
        sizes = {}
        for table in schema.tables:
            sizes[table.table_name] = table.table_size
    
        n_cust = sizes.get('customer', 150000)
        n_part = sizes.get('part', 600000)
        n_supp = sizes.get('supplier', 10000)
        print(f"Reference Table Sizes: Customer={n_cust}, Part={n_part}, Supplier={n_supp}")
    except Exception as e:
        print(f"Error getting schema info: {e}. Using default SF5 sizes.")
        n_cust = 150000
        n_part = 600000
        n_supp = 10000

    for skew in skew_params:
        print(f"\n--- Generating data for skew={skew} ---")
        dst_dir = os.path.join(dst_base_path, f"skew_{skew}")
        os.makedirs(dst_dir, exist_ok=True)
        
        dims = ['customer.csv', 'part.csv', 'supplier.csv', 'date.csv']
        for dim in dims:
            src = os.path.join(src_path, dim)
            dst = os.path.join(dst_dir, dim)
            if os.path.exists(src):
                if not os.path.exists(dst):
                    print(f"Copying {dim}...")
                    shutil.copy2(src, dst)
                else:
                    print(f"{dim} already exists.")
            else:
                print(f"Warning: {src} not found!")

        # lineorder
        lo_src = os.path.join(src_path, 'lineorder.csv')
        lo_dst = os.path.join(dst_dir, 'lineorder.csv')
        
        if not os.path.exists(lo_src):
             print(f"Error: {lo_src} not found!")
             continue
             
        if os.path.exists(lo_dst):
            print(f"lineorder.csv for skew={skew} already exists, skipping...")
            continue

        print(f"Processing lineorder.csv...")
        chunksize = 1000000
        first = True
        date_keys = None # Cache for date keys
        
        # We assume headers exist
        try:
            reader = pd.read_csv(lo_src, chunksize=chunksize)
            
            for chunk in tqdm(reader):
                n_rows = len(chunk)
                
                # Check for columns and apply skew
                if 'lo_custkey' in chunk.columns:
                    chunk['lo_custkey'] = generate_bounded_zipfian(n_cust, skew, n_rows)
                
                if 'lo_partkey' in chunk.columns:
                    chunk['lo_partkey'] = generate_bounded_zipfian(n_part, skew, n_rows)
                    
                if 'lo_suppkey' in chunk.columns:
                    chunk['lo_suppkey'] = generate_bounded_zipfian(n_supp, skew, n_rows)
                
                if 'lo_orderdate' in chunk.columns:
                    if date_keys is None:
                        try:
                            date_df = pd.read_csv(os.path.join(src_path, 'date.csv'))
                            possible_keys = ['d_datekey', 'D_DATEKEY', 'datekey']
                            key_col = next((k for k in possible_keys if k in date_df.columns), None)
                            if key_col:
                                date_keys = date_df[key_col].values
                                n_date = len(date_keys)
                                print(f"Loaded {n_date} date keys for skewing lo_orderdate.")
                            else:
                                print("Warning: Could not find date key column in date.csv. Skipping lo_orderdate skew.")
                                date_keys = [] 
                        except Exception as e:
                            print(f"Error loading date keys: {e}. Skipping lo_orderdate skew.")
                            date_keys = []

                    if len(date_keys) > 0:
                        # Generate indices 0..N-1 with skew
                        skewed_indices = generate_bounded_zipfian(len(date_keys), skew, n_rows) - 1
                        # Map back to actual date keys
                        chunk['lo_orderdate'] = date_keys[skewed_indices]
                
                mode = 'w' if first else 'a'
                header = first
                chunk.to_csv(lo_dst, index=False, header=header, mode=mode)
                first = False
                
        except Exception as e:
            print(f"Failed to process lineorder.csv: {e}")

    print("\nData generation complete.")

def generate_skewed_ssb_partname_only(
    src_path,
    dst_base_path,
    skew_params=[0.0, 0.5, 1.0, 1.5, 2.0],
):
    def _make_zipf_sampler(n, alpha):
        if alpha == 0:
            return lambda size: np.random.randint(0, n, size)

        k = np.arange(1, n + 1, dtype=np.float64)
        weights = 1.0 / (k ** alpha)
        cum_probs = np.cumsum(weights)
        cum_probs /= cum_probs[-1]

        def _sample(size):
            u = np.random.random(size)
            return np.searchsorted(cum_probs, u)

        return _sample

    def _estimate_zipf_alpha_from_counts(counts):
        freqs = counts[counts > 0].astype(float)
        freqs.sort()
        freqs = freqs[::-1]
        ranks = np.arange(1, len(freqs) + 1, dtype=float)

        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs)
        slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
        alpha_est = -slope

        y_pred = slope * log_ranks + intercept
        ss_res = np.sum((log_freqs - y_pred) ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

        total = freqs.sum()
        top1_share = freqs[0] / total
        top10_share = freqs[:10].sum() / total

        return alpha_est, r2, top1_share, top10_share

    print("Generating skewed SSB datasets (p_name only).")
    print(f"Source: {src_path}")
    print(f"Destination Base: {dst_base_path}")

    part_src = os.path.join(src_path, "part.csv")
    if not os.path.exists(part_src):
        raise FileNotFoundError(f"part.csv not found at {part_src}")

    tname = "part"
    part_df = pd.read_csv(part_src, names=get_ssb_colname(tname), header=None, sep='|', usecols=usefulCols_ssb(tname))
    possible_cols = ["p_name", "P_NAME", "p_partname", "P_PARTNAME"]
    name_col = next((c for c in possible_cols if c in part_df.columns), None)
    if not name_col:
        raise ValueError("Could not find p_name column in part.csv")

    unique_names = part_df[name_col].unique()
    n_names = len(unique_names)
    print(f"Found {n_names} unique values in {name_col}.")

    for skew in skew_params:
        print(f"\n--- Generating data for skew={skew} ---")
        dst_dir = os.path.join(dst_base_path, str(skew))
        os.makedirs(dst_dir, exist_ok=True)

        # Copy all tables unchanged first
        for filename in os.listdir(src_path):
            if not filename.endswith(".csv"):
                continue
            src = os.path.join(src_path, filename)
            dst = os.path.join(dst_dir, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            else:
                print(f"{filename} already exists.")

        # Overwrite part.csv with skewed p_name values
        n_rows = len(part_df)
        sampler = _make_zipf_sampler(n_names, skew)
        skewed_indices = sampler(n_rows)
        skewed_names = unique_names[skewed_indices]
        skewed_part_df = part_df.copy()
        skewed_part_df[name_col] = skewed_names

        skewed_part_df.to_csv(
            os.path.join(dst_dir, "part.csv"),
            index=False,
            header=False,
            sep="|",
        )
        print(f"Updated part.csv with skewed {name_col}.")
        counts = np.bincount(skewed_indices, minlength=n_names)
        alpha_est, r2, top1, top10 = _estimate_zipf_alpha_from_counts(counts)
        print(
            "Skew check "
            f"(expected alpha={skew}, estimated={alpha_est:.3f}, r2={r2:.3f}, "
            f"top1={top1:.3%}, top10={top10:.3%})"
        )
        if skew > 0 and abs(alpha_est - skew) > 0.2:
            print("Warning: estimated skew differs from target by > 0.2")

    print("\nData generation complete.")

if __name__ == "__main__":
    src = os.path.expanduser("~/Desktop/data/ssb/sf50")
    dst = os.path.expanduser("~/Desktop/data/ssb/sf50-skew")
    generate_skewed_ssb_partname_only(src, dst)
