import numpy as np
from server.tensor_server import load_tensors
import sparse
import os
import sys
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

def compute_sparsity(arr):
    if not isinstance(arr, np.ndarray):
        if isinstance(arr, sparse.COO):
            arr = arr.todense()
        arr = np.array(arr)
        
    return (1-np.count_nonzero(arr)/np.prod(arr.shape))*100

if __name__ == "__main__":
    bin_size = 1000
    avg_sparsity = dict()
    cpds = load_tensors(f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_dense.npz', f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_sparse.pkl')
    for k, v in cpds.items():
        if k.startswith('lo'):
            if k.endswith('_m') or k.endswith('_m_') or k.endswith('_b2v') or k.endswith('_b2v_'):
                continue
            else:
                if "lineorder" not in avg_sparsity:
                    avg_sparsity["lineorder"] = []
                avg_sparsity["lineorder"].append(compute_sparsity(v))
        elif k.startswith('c'):
            if k.endswith('_m') or k.endswith('_m_') or k.endswith('_b2v') or k.endswith('_b2v_'):
                continue
            else:
                if "customer" not in avg_sparsity:
                    avg_sparsity["customer"] = []
                avg_sparsity["customer"].append(compute_sparsity(v))
        elif k.startswith('p'):
            if k.endswith('_m') or k.endswith('_m_') or k.endswith('_b2v') or k.endswith('_b2v_'):
                continue
            else:
                if "part" not in avg_sparsity:
                    avg_sparsity["part"] = []
                avg_sparsity["part"].append(compute_sparsity(v))
        elif k.startswith('d'):
            if k.endswith('_m') or k.endswith('_m_') or k.endswith('_b2v') or k.endswith('_b2v_'):
                continue
            else:
                if "dwdate" not in avg_sparsity:
                    avg_sparsity["dwdate"] = []
                avg_sparsity["dwdate"].append(compute_sparsity(v))
        elif k.startswith('s'):
            if k.endswith('_m') or k.endswith('_m_') or k.endswith('_b2v') or k.endswith('_b2v_'):
                continue
            else:
                if "supplier" not in avg_sparsity:
                    avg_sparsity["supplier"] = []
                avg_sparsity["supplier"].append(compute_sparsity(v))

    for k, v in avg_sparsity.items():
        print(f"{k}: {np.mean(v)}")