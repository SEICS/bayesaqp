import pickle
import socket
import sys
import os

# # Add the project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

import numpy as np
import traceback
import ast
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy.core")

try: # for module import
    from .tensor_loader import tensor_load
    from .tensor_exec import tensor_exec
except ImportError:
    # run as python server/tensor_server.py
    from tensor_loader import tensor_load
    from tensor_exec import tensor_exec
import opt_einsum
from opt_einsum import contract, shared_intermediates, RandomGreedy
import math
import re
import time
from itertools import chain
from time import perf_counter
import sparse
import torch
import os
import argparse

os.environ["SPARSE_BACKEND"] = "Finch"
# torch.backends.opt_einsum.enabled = False
# torch.backends.opt_einsum.strategy = 'dp'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Pytorch: If GPU is available, use GPU, otherwise use CPU

def force_cpu(arr):
	if isinstance(arr, torch.Tensor):
		return arr.cpu().numpy()
	return arr

def interpret_coo_sparsity(x):
	if isinstance(x, sparse.COO):
		nnz = x.nnz
		total = np.prod(x.shape)
		density = nnz / total
		return 1.0 - density

def torch_arr(x):
	if isinstance(x, sparse.COO):
		# if interpret_coo_sparsity(x) > 0.67:
		# 	t = torch.sparse_coo_tensor(x.coords, x.data, x.shape, dtype=torch.float32)
		# else:	
		t = torch.from_numpy(x.todense())
	elif isinstance(x, (float, int)):
		t = torch.tensor(x, dtype=torch.float32)
	elif isinstance(x, list):
		t = torch.tensor(x, dtype=torch.float32)
	elif isinstance(x, np.ndarray):
		# Skip object arrays that contain non-numerical data
		if x.dtype == np.object_:
			# Check if array contains strings or other non-numerical objects
			has_strings = any(isinstance(val, str) for val in x.flatten())
			has_objects = any(not isinstance(val, (int, float, np.number)) for val in x.flatten())
			
			if has_strings or has_objects:
				raise TypeError(f"Skipping object array with non-numerical data: {type(x)}")
			else:
				# All values are numerical, convert to float64 then to tensor
				x_converted = x.astype(np.float64)
				t = torch.from_numpy(x_converted)
		else:
			# Regular numerical array
			t = torch.from_numpy(x)
	elif isinstance(x, torch.Tensor):
		t = x
	else:
		raise TypeError(f"Unsupported type: {type(x)}")
	
	# if device.type == 'mps':
	# 	if t.is_sparse:
	# 		t = t.to_dense()
	# 	return t.float().half().to(device)
	
	return t.float().to(device) # torch requires same dtype and device(GPU/CPU) for all tensors. Mps/mlx doesn't support float64/double()

def get_true_names(dict):
    return list(chain.from_iterable(dict.values()))

def get_true_group_by_combine(gbt_lst, idx_matching):
    return tuple([gbt_lst[i][attr] for (i, attr) in enumerate(idx_matching[:-1])])

def reverse_map(res, *gb): # general
	# print(f"result sparsity before reversemap: {compute_sparsity(res)*100}%")
	if isinstance(res, torch.Tensor):
		res = res.cpu().numpy() # keep everything on CPU and in numpy afterwards
		
	idx = np.nonzero(res)
	rng = list(zip(*idx, res[idx]))
	gbt_lst = [get_true_names(g) for g in gb]	
	res_tup = [(get_true_group_by_combine(gbt_lst, idx_matching), idx_matching[-1]) for idx_matching in rng]
	return dict(res_tup)

def load_tensors(dense_path, sparse_path): # Don't need to convert sparse to dense. It kills the process, convert sparse to dense using too much RAM
	dense_data = dict(np.load(dense_path, allow_pickle=True)) # load dense tensor
	# dense_data = dict([(k, np.ascontiguousarray(v)) for k, v in dense_data.items()])

	with open(sparse_path, "rb") as f:
		sparse_data_raw = pickle.load(f) # load sparse tensor
	env = {**dense_data, **sparse_data_raw}
	return env

def take_tensor(t, idx, axis=0):
	if isinstance(t, np.ndarray):
		return np.take(t, idx, axis=axis)

	elif isinstance(t, torch.Tensor):
		idx = torch.tensor(idx, device=t.device, dtype=torch.long)
		return torch.index_select(t, dim=axis, index=idx)

	elif isinstance(t, sparse.COO):
		axis_coords = t.coords[axis]
		idx = np.asarray(idx)

		max_idx = axis_coords.max()
		lookup = np.zeros(max_idx + 1, dtype=bool) 
		lookup[idx] = True
		mask = lookup[axis_coords]

		# Remap coordinates
		remap = np.full(max_idx + 1, -1, dtype=int)
		remap[idx] = np.arange(len(idx))
		new_coords = t.coords[:, mask].copy()
		new_coords[axis] = remap[axis_coords[mask]]

		new_shape = list(t.shape)
		new_shape[axis] = len(idx)
		return sparse.COO(coords=new_coords, data=t.data[mask], shape=tuple(new_shape)).todense()

	else:
		raise TypeError("Unsupported tensor type")

def compute_sparsity(arr):
    if not isinstance(arr,np.ndarray):
        if isinstance(arr, sparse.COO):
            arr = arr.todense()
        arr = np.array(arr)
        
    return 1-np.count_nonzero(arr)/np.prod(arr.shape) # np.product is deprecated

def densify_sparse(t):
    if isinstance(t, sparse.COO):
        return t.todense() # return numpy.ndarray
    else:
        return t
    
def recvall(sock):
    BUFF_SIZE = 4096 * 1024 # 4 MB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data
END_MARK = '<END>"'
END_MARK2 = "<END>'"
END_SIZE = len(END_MARK)
LOAD_CMD = 'LOAD'
def recvfull(sock):
	out = b''

	while True:
		line = recvall(sock)
		sline = str(out + line)
		print('received <%d>' % len(sline), file=sys.stderr)
		if(sline.endswith(END_MARK) or sline.endswith(END_MARK2)):
			out += line
			out = out[0:-END_SIZE]
			break
		if(line == b''):
			break
		out += line
		pass
	return out

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--backend', default='numpy', help='Which backend to be used (either numpy or torch)')
	# parser.add_argument('--dataset', default='flights', help='Which dataset statistics to be loaded')
	# parser.add_argument('--scale', default='1B', help='Which scale to be used') # 
	args = parser.parse_args()
	# print(f"sys.argv: {sys.argv}")
	if (len(sys.argv) > 3) or len(sys.argv) < 1: # backend is optional, by default, use numpy
		print('Usage: python tensor_server.py [--backend numpy|torch]')
		sys.exit(1)
	conf_file = open('server/conf.json')
	conf_str = conf_file.read()
	conf_data = json.loads(conf_str)
	port = conf_data['port']
	address = conf_data['address']
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_address_port = (address, port)
	print('starting up on %s port %s' % server_address_port, file=sys.stderr)
	sock.bind(server_address_port)
	sock.listen(1)

	# TODO: Need to load the cpds for the different datasets and scales with argparse
	bin_size = 1000
	flights_cpds = dict(np.load('progs/aqp/flights/cpds/test/bin1500.npz', allow_pickle=True)) # should be passed by user with argparse
	# synthetic_cpds = dict(np.load('progs/aqp/synthetic/cpds/change_s/s0.0_c0.4_dom10000/bin1500.npz', allow_pickle=True))

 
	ssb_cpds = load_tensors("progs/aqp/ssb/cpds/sf50-test/bin1000_dense.npz", "progs/aqp/ssb/cpds/sf50-test/bin1000_sparse.pkl")
	# ssb_cpds = load_tensors(f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_dense.npz', f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_sparse.pkl') # for sparse tensor
	# ssb_cpds = load_tensors(f'progs/aqp/ssb/cpds/sf50-npcomressed/bin{bin_size}_dense.npz', f'progs/aqp/ssb/cpds/sf50-npcomressed/bin{bin_size}_sparse.pkl') # for sparse tensor
	# print(f"ssb_cpds keys: {ssb_cpds.keys()}")

	# # skewness on group by p_name:
	# skewness = "sf50-skew15"
	# ssb_cpds = load_tensors(f"progs/aqp/ssb/cpds/{skewness}/bin1000_dense.npz", f"progs/aqp/ssb/cpds/{skewness}/bin1000_sparse.pkl")


	# Need to manually load the cpds for the ablation study for now
	# param = 8
	# uniqValGB_cpds = load_tensors(f'progs/plots/ssb/uniqValGB/sf50/cpds/{param}k/bin{bin_size}_dense.npz', f'progs/plots/ssb/uniqValGB/sf50/cpds/{param}k/bin{bin_size}_sparse.pkl') # for sparse tensor
	# ssb_cpds["part_p_name_b2v"] = uniqValGB_cpds["part_p_name_b2v"]
	# ssb_cpds["p_name"] = uniqValGB_cpds["p_name"]
	# ssb_cpds["lo_partkey_p_partkey"] = uniqValGB_cpds["lo_partkey_p_partkey"]
	# ssb_cpds["part_p_name"] = uniqValGB_cpds["part_p_name"]
	# # ssb_cpds.update(uniqValGB_cpds) # Bug: cause performance degradation
	# # print(f"finished updating ssb_cpds")
	# # # print(f"ssb_cpds keys: {ssb_cpds['part_p_name_b2v']}")	

	# # print(f"ssb_cpds keys: {ssb_cpds.keys()}")	
	# for AQP on imdb
	imdb_cpds = load_tensors("progs/aqp/imdb/cpds/bin1000_dense.npz", "progs/aqp/imdb/cpds/bin1000_sparse.pkl") # for testing progs/update_join/sql/alltests.sql
	# for chain join testing on imdb origin
	# imdb_cpds = load_tensors("progs/update_join/imdb-origin/cpds/bin1000_dense.npz", "progs/update_join/imdb-origin/cpds/bin1000_sparse.pkl")
	# for chain join testing on imdb zero, ci has zero correlation with any other table except t
	# imdb_cpds = load_tensors("progs/update_join/imdb-zero/cpds/bin1000_dense.npz", "progs/update_join/imdb-zero/cpds/bin1000_sparse.pkl")

	for (k, v) in imdb_cpds.items():
		print(f"k: {k}, v: {v.shape}")
	# print(f"synthetic_cpds keys: {synthetic_cpds.keys()}")
	print(f"Finished loading the cpds.")

	if args.backend == 'torch':
		# Combine dictionaries first
		combined_cpds = {**flights_cpds, **ssb_cpds, **imdb_cpds}
		# combined_cpds = {**imdb_cpds}#, **synthetic_cpds}
		for (k, v) in combined_cpds.items():
			if v is None:
				continue
			
			if k.endswith("b2v"): # skip b2v (vars for reverse map) because it might contains other dtypes and mps only supports numercial computations
				globals()[k] = v
			else:
				try:
					globals()[k] = torch_arr(v)
				except (TypeError, ValueError) as e:
					print(f"Warning: Skipping {k} - {e}")
					globals()[k] = v  # Keep original value
	else: # by default, execute in numpy
		globals().update(flights_cpds)
		globals().update(ssb_cpds)
		globals().update(imdb_cpds)
		# globals().update(synthetic_cpds)
	while True:
		connection, client_address = sock.accept()
		print('connection from ', client_address, file=sys.stderr)
		try:
			line = recvfull(connection)
			outputLine = ''
			warmup = 4
			runs = 5
			avg_lat = []
			try:
				if(line.decode().startswith(LOAD_CMD)):
					file_name = line.decode()[len(LOAD_CMD):].strip()[1:-1]
					tensor_load(file_name, globals())
					res = 'LOADED %s' % file_name
				else:
					total_lat = 0
					
					# Since we cannot modify code generation, we just log progress without query ID
					# The user can correlate timestamps in logs with their execution script if needed.
					
					# Warmup runs
					for i in range(warmup):
						print(f"Warmup run {i+1}/{warmup}", file=sys.stderr)
						tensor_exec(line.decode(), globals())
					
					# Measured runs
					for i in range(runs):
						print(f"Measured run {i+1}/{runs}", file=sys.stderr)
						res = tensor_exec(line.decode(), globals())	
						total_lat += res[0]
					
					avg_lat = total_lat / runs
					print(f"Average latency over {runs} runs: {avg_lat:.3f} ms\n", file=sys.stderr)
					# Use the result (res) from the last run
					res = (avg_lat, res[1])

				res_dir = "server/res/" 
				if not os.path.exists(res_dir):
					os.makedirs(res_dir) 
				npy_path = os.path.join(res_dir, "result_rev_auto.npy")
				np.save(open(npy_path, "wb"), res[1], allow_pickle=True)
				
				# Log latency and result info to JSONL
				log_entry = {
					"timestamp": time.time(),
					"latency": res[0],
					"result_path": npy_path
				}
				with open(f"server/server_log_{args.backend}.jsonl", "a") as log_file:
					log_file.write(json.dumps(log_entry) + "\n")
					log_file.flush()

				# print(res)
				# if avg_lat:
				# 	lat = np.mean(avg_lat)

				# outputLine += f"({res[0]},)" + '\n' # only report latency
				outputLine += f"({res[0]}, {str(res[1])})" + '\n' # report latency and result
				# outputLine += str(lat) + '\n'
				connection.sendall(outputLine.encode())
				print('Sent: ' + outputLine)
				# print('Avgerage latency: ' + outputLine)
				# print(f'Inference_time: {inference_time}ms.')
			except Exception as e:
				traceback.print_exc()
				print("[INTERNAL ERROR] ", e, file=sys.stderr)
			pass
		except Exception as e:
			traceback.print_exc()
			print("[ERROR] ", e, file=sys.stderr)
		finally:
			connection.close()
	sock.close()