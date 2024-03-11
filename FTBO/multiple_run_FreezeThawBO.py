import os

cpu_lim = 2
os.environ['OMP_NUM_THREADS'] = str(cpu_lim)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_lim)
os.environ['MKL_NUM_THREADS'] = str(cpu_lim)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_lim)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_lim)

import gzip
import pickle
import json
import torch
import scipy
from src.ei import EI
from src.cmaes import CMAES
import numpy as np
from src.freeze_thaw_bayesian_optimization import FreezeThawBO
from src.freeze_thaw_model import FreezeThawGP
from src.direct import Direct
from TopoOpt_weibo import TopoOpt
import sys
import time

argv = sys.argv[1:]
with open(argv[0]) as f:
    conf = json.load(f)

outdim = 5
inputdim=10

max_fun = conf['max_fun']
init_points = conf['init_points']
circuit_dir = conf['circuit_dir']
DVAE_model_dir = conf['DVAE_model_dir']

basket_size = 3
threshold = 1
n_std = 1.5
n_bfgs = 10
# pkl_dir = "./pickle_b{}_th{}_i{}_r{}_bound{}_bfgs{}".format(basket_size, threshold, init_points, max_fun, n_std, n_bfgs)
pkl_dir = "./pickle"
if not os.path.exists(pkl_dir):
    os.makedirs(pkl_dir)

device = torch.device('cpu')
DVAE_model = torch.load(os.path.join(DVAE_model_dir, 'model_checkpoint500.pth'), map_location=device).to(device)
bounds = np.zeros((2, inputdim))
bounds[0] = -n_std * np.ones(inputdim)
bounds[1] = n_std * np.ones(inputdim)

times = 10
for i in range(times):
    time_start = time.time()
    TopoOptTask = TopoOpt(bounds, outdim, DVAE_model, circuit_dir)
    bo = FreezeThawBO(task=TopoOptTask, init_points=init_points, basket_size=basket_size, threshold=threshold, spec=outdim, n_bfgs=n_bfgs)

    incumbent, incumbent_value, time_dict, res_list = bo.run(max_fun)
    with open(os.path.join(pkl_dir, 'res_{}.pkl'.format(i)), 'wb') as f1:
        pickle.dump(res_list, f1)
    print("run ", i + 1, " FINAL incumbent = ", incumbent, " value = ", incumbent_value)
    time_end = time.time()
    topo_hist = TopoOptTask.topo_hist
    time_dict['total_time'] = time_end - time_start
    with open(os.path.join(pkl_dir, 'topo_hist_{}.pkl'.format(i)), 'wb') as f:   
        pickle.dump(topo_hist, f)
    with open(os.path.join(pkl_dir, 'time_dict_{}.pkl'.format(i)), 'wb') as f1:
        pickle.dump(time_dict, f1)
