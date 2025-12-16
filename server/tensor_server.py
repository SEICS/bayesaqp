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

def torch_arr(x):
	if isinstance(x, sparse.COO):
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
	parser.add_argument('--backend', default='numpy', help='Which backend to be used')
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
	# flights_cpds = dict(np.load('progs/aqp/flights/cpds/1B/bin1500.npz', allow_pickle=True)) 
	# flights_cpds = dict(np.load('progs/aqp/flights/cpds/1B/cpds/bin1500.npz', allow_pickle=True))
 
	ssb_cpds = load_tensors("progs/aqp/ssb/cpds/sf50-test/bin1000_dense.npz", "progs/aqp/ssb/cpds/sf50-test/bin1000_sparse.pkl")
	# ssb_cpds = load_tensors(f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_dense.npz', f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_sparse.pkl') # for sparse tensor
	# ssb_cpds = load_tensors(f'progs/aqp/ssb/cpds/sf50-npcomressed/bin{bin_size}_dense.npz', f'progs/aqp/ssb/cpds/sf50-npcomressed/bin{bin_size}_sparse.pkl') # for sparse tensor
	# # print(f"ssb_cpds keys: {ssb_cpds.keys()}")

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
	print(f"Finished loading the cpds.")

	if args.backend == 'torch':
		# Combine dictionaries first
		combined_cpds = {**flights_cpds, **ssb_cpds}
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
					# for i in range(warmup+runs+1):
					res = tensor_exec(line.decode(), globals())	
					# print(f"res: {res}")
						# if i >= warmup:
						# 	avg_lat.append(res[0])
				res_dir = "server/res/" 
				if not os.path.exists(res_dir):
					os.makedirs(res_dir)
				file_path = np.save(open(os.path.join(res_dir, "result_rev_auto.npy"), "wb"), res[1], allow_pickle=True)
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