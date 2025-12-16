import numpy as np
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
    cpds = dict(np.load('progs/aqp/flights/cpds/1B/cpds/bin1500.npz', allow_pickle=True))
    for k, v in cpds.items():
        print(f"{k}: {compute_sparsity(v)}")