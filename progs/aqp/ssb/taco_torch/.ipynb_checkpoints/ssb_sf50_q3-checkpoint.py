for (k,v) in data.items():
...     if v.dtype == "object":
...             rmap[k] = v
...     else:
...             if isinstance(v, sparse.COO):
...                     taco[k] = pt.from_array(v.todense())
...             else:
...                     taco[k] = pt.from_array(v)