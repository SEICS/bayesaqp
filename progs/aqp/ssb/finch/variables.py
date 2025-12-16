import sys
import os
import sparse
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

from server.tensor_server import load_tensors

bin_size = 1000
ssb_cpds = load_tensors(f'progs/aqp/ssb/cpds/sf50/bin1000_dense.npz', f'progs/aqp/ssb/cpds/sf50/bin1000_sparse.pkl') # for sparse tensor