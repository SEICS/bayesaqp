import json
import sys
import os
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

import numpy as np
import binsparse
from variables import ssb_cpds
import sparse
import zarr
import h5py
import scipy
import finch
from io import BytesIO
from scipy.sparse import coo_array
from scipy.io import mmwrite

globals().update(ssb_cpds)

def binsparse_store():
    # for (k, v) in ssb_cpds.items():
    #     print(f"{k} type: {type(v)}")
    # # supplier_s_suppkey = scipy.sparse.coo_matrix(supplier_s_suppkey.todense())

    # z = zarr.group()
    # binsparse.write(z, "customer_c_custkey", customer_c_custkey)
    # binsparse.read(z["csr"])


    h5_cpds = dict()
    with h5py.File("progs/aqp/ssb/finch/cpds.bsp.h5", "w") as f:
        for (k, v) in ssb_cpds.items():
            if isinstance(v, np.ndarray):
                # binsparse.write(f, k, v)
                continue
            else:
                # h5_cpds[k] = scipy.sparse.coo_matrix(v.todense())
                # vd = np.array(v.todense())
                spt = scipy.sparse.csc_matrix(v.todense())
                binsparse.write(f, k, spt)   # Writes into file
    f.close()

    with h5py.File("progs/aqp/ssb/finch/cpds.bsp.h5", "r") as f1:
        vars = list(f1.keys())
        print("what variables have been stored:", f1.name)  # should show ["A"]
        g = f1[vars[0]]                                  # the group that holds the tensor
        cpds = binsparse.read(g)
        print(f"{vars[0]}: {type(cpds), cpds.shape}")
    f1.close()

    # import numpy as np
    # from scipy import sparse

    # import zarr

    # csr = sparse.random(1000, 100, format="csr", density=0.1, random_state=np.random.default_rng())
    # z = zarr.group()

    # binsparse.write(z, "csr", csr)
    # binsparse.read(z["csr"])

def mtx_store(path): # each .mtx file is one matrix
    data_path = os.path.join(path, "data") 
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    dense = dict()
    for (k, v) in ssb_cpds.items():
        # print(f"{k}, {type(v)}")
        if isinstance(v, sparse._coo.core.COO): # each sparse matrix is stored as one .mtx
            print(f"{k}, {v.dtype}, {type(v)}")
            mmwrite(f"{data_path}/{k}.mtx", v.to_scipy_sparse()) # converted from sparse.COO to scipy.coo_matrix
        elif isinstance(v, np.ndarray):
            dense[k] = v
        
    np.savez(f"{data_path}/dense.npz", **dense)
    return None 

if __name__ == "__main__":
    path = "progs/aqp/ssb/finch/"
    # binsparse_store()
    mtx_store(path)