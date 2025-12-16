import h5py
import binsparse
from scipy import sparse

coo = sparse.random(10, 5, density=0.2, format="coo", random_state=0)

with h5py.File("tensor.bsp.h5", "w") as f:
    binsparse.write(f, "A", coo)   # Writes into file

# with h5py.File("tensor.bsp.h5", "r") as f1:
#     print("top-level groups:", list(f1.keys()))  # should show ["A"]
#     g = f1["A"]                                  # the group that holds the tensor
#     cpds = binsparse.read(g)
#     print(type(cpds), cpds.shape)