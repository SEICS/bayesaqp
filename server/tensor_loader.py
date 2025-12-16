import numpy as np
import ast

def tensor_load(file_name, context):
	with open(file_name, 'r') as content_file:
		content = content_file.read()
		tree = ast.parse(content)
		expr = ast.Module(tree, type_ignores=[])
		# print(f"expr: {expr}")
		# print(f"context: {context}")
		exec(compile(expr.body, 'file', 'exec'), context)
		# print(wf_c_region_weight)

# tensor_load('server/load_sample.py')

# print(wf_c_region_weight)
# print(lineorder_idx_mat_lo_suppkey)

