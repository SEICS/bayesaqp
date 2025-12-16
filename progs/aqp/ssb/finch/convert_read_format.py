import sys
import os
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

import numpy as np
import sparse 

from variables import ssb_cpds
# globals().update(ssb_cpds)

def convert_into_npz(path):
    for (attr, val) in ssb_cpds.items():
        if isinstance(val, sparse.COO):
            np.savez_compressed(
                path,
                kind="coo",
                indices=val.coords,
                values=val.data,
                shape=val.shape,
                dtype=str(val.dtype),
            )

# Tensor converted should be able to read in julia via .npz/.json files, and easily converted to Finch.Tensor, e.g., reading A, then T = Tensor(COOFormat(), A)
if __name__ == "__main__":
    convert_into_npz()