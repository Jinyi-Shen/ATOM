import os
import autograd.numpy as np
import string
import torch
import time
from netlist_generator import *
from src.util import *
from src.BO import BO
from src.base_task import BaseTask
from scipy.stats import qmc


class TopoOpt(BaseTask):

    def __init__(self, bounds, outdim, DVAE_model, circuit_dir):
        self.base_name = "TopoOpt"
        self.out_dim = outdim
        self.DVAE_model = DVAE_model
        self.circuit_dir = circuit_dir
        self.topo_hist = {}
        self.sizing_num = 5
        self.sizing_iter = 10
        X_lower = np.array(bounds[0])
        X_upper = np.array(bounds[1])
        super(TopoOpt, self).__init__(X_lower, X_upper)

    def decode_topo(self, x):
        device = torch.device('cpu')
        x0 = torch.from_numpy(x.reshape((1, -1))).float().to(device)
        g_recon = self.DVAE_model.decode(x0, stochastic=False)[0]

        block_type_list = []
        for i in range(1, 11):
            block_type_list.append(int(g_recon.vs[i]['type']))
        topo = np.zeros(8).astype(int)
        if block_type_list[0] == 7 and block_type_list[1] == 8 and block_type_list[7] == 7:
            topo[0] = 1
        elif block_type_list[0] == 8 and block_type_list[1] == 7 and block_type_list[7] == 8:
            topo[0] = 0
        else:
            raise NotImplementedError
        if block_type_list[2]:
            topo[1] = 1
        else:
            topo[1] = 0
        if block_type_list[3]:
            topo[2] = 1
        else:
            topo[2] = 0
        if block_type_list[4]:
            topo[3] = 1
        else:
            topo[3] = 0
        topo[4] = block_type_list[5]
        topo[5] = block_type_list[6]
        topo[6] = block_type_list[8]
        topo[7] = block_type_list[9]

        self.topo = tuple(topo)
        return '_'.join(map(str, self.topo))

    def objective_function(self):
        result = np.zeros(self.out_dim)
        topo = self.topo
        amp_generator(topo, self.circuit_dir)
        print('########################################evaluating topo', topo)
        if topo in self.topo_hist.keys():
            sizing_bounds = self.topo_hist[topo]['sizing_bounds']
            dataset = self.topo_hist[topo]['dataset']
        else:
            block_type_list = topo2block(topo)
            sizing_bounds = gen_sizing_bounds(block_type_list)
            num = self.sizing_num
            dataset = self.init_dataset_simulate(simulate, num, sizing_bounds)
            dataset['gp_time'] = []
            dataset['opt_time'] = []
            dataset['time'] = []

        gp_time = 0
        opt_time = 0
        K = 70
        iteration = self.sizing_iter
        time_start0 = time.time()
        for ii in range(iteration):
            time_start = time.time()
            model = BO(dataset)
            time_end1 = time.time()

            def task(x0):
                x0 = model.optimize_tt_constr(x0)
                x0 = model.optimize_tt_wEI(x0)
                wEI_tmp = model.calc_log_wEI_approx(x0)
                return x0, wEI_tmp

            x0_list = []
            x0_list.append(model.rand_x(K))
            results = list(map(task, x0_list))

            candidate = results[0][0]
            wEI_tmp = results[0][1]

            idx = np.argsort(wEI_tmp)[-1:]
            time_end = time.time()
            new_x = candidate[idx]
            new_y = simulate(new_x, sizing_bounds)
            gp_time += time_end1 - time_start
            opt_time += time_end - time_end1
            dataset['train_x'] = np.concatenate((dataset['train_x'], new_x))
            dataset['train_y'] = np.concatenate((dataset['train_y'], new_y))

        time_end0 = time.time()
        dataset['time'].append(time_end0 - time_start0)
        dataset['gp_time'].append(gp_time)
        dataset['opt_time'].append(opt_time)

        topo_item = {}
        topo_item['sizing_bounds'] = sizing_bounds
        topo_item['dataset'] = dataset
        self.topo_hist[topo] = topo_item
        best_y = model.best_y
        result[0] = best_y[0] + 10 * np.sum(np.maximum(best_y[1:], 0))
        result[1:] = best_y[1:]
        return result

    def init_dataset_simulate(self, funct, num, bounds):
        dim = bounds.shape[1]
        sampler = qmc.LatinHypercube(d=dim, seed=1)
        sample = sampler.random(num)
        lb = np.zeros(dim)
        ub = np.ones(dim)
        xh = qmc.scale(sample, lb, ub)
        x = np.asarray(xh)
        dataset = {}
        dataset['train_x'] = x
        dataset['train_y'] = funct(x, bounds)
        return dataset


def convert(s):
    if s[-1] == 'm' or s[-1] == 'M':
        v = float(s[:-1]) * 1e-3
    elif s[-1] == 'u' or s[-1] == 'U':
        v = float(s[:-1]) * 1e-6
    elif s[-1] == 'n' or s[-1] == 'N':
        v = float(s[:-1]) * 1e-9
    elif s[-1] == 'p' or s[-1] == 'P':
        v = float(s[:-1]) * 1e-12
    elif s[-1] == 'k' or s[-1] == 'K':
        v = float(s[:-1]) * 1e3
    elif s[-1] == 'g' or s[-1] == 'G':
        v = float(s[:-1]) * 1e9
    elif s[-1] == 'x' or s[-1] == 'X':
        v = float(s[:-1]) * 1e6
    elif s[-1] == 'f' or s[-1] == 'F':
        v = float(s[:-1]) * 1e-15
    else:
        v = float(s)
    return v


def gen_sizing_bounds(block_type_list):
    bound_g = [1e-5, 1e-3] #spec1
    bound_r_prs = [40, 80] #spec1
    bound_r = [1e4, 1e6] #spec1
    bound_c1 = [5e-13, 5e-12] #spec1
    #bound_g = [1e-5, 1e-2] #spec2
    #bound_r_prs = [40, 80] #spec2
    #bound_r = [1e5, 1e6] #spec2
    #bound_c1 = [1e-13, 1e-11] #spec2
    bound_c3 = [5e-13, 1e-12]

    bounds_list = []

    for i, x in enumerate(block_type_list):
        if x == 2:  # C
            bounds_list.append(bound_c1)
        elif x == 1:  # RC series
            bounds_list.append(bound_r)
            bounds_list.append(bound_c1)
        elif x == 5 or x == 6:
            if i == 0 or i == 1 or i == 2:
                bounds_list.append(bound_g)
                bounds_list.append(bound_r_prs)
            else:
                bounds_list.append(bound_g)
                bounds_list.append(bound_r_prs)
                bounds_list.append(bound_c3)
        elif x == 3 or x == 4:
            bounds_list.append(bound_g)
            bounds_list.append(bound_r_prs)
            bounds_list.append(bound_c3)
            bounds_list.append(bound_c1)
    bounds = np.array(bounds_list).T
    return bounds


def topo2block(topo):
    if topo[0]:
        block_type_list = [5, 6, 5]
        if topo[1]:
            block_type_list.append(5)
        if topo[2]:
            block_type_list.append(6)
        if topo[3]:
            block_type_list.append(5)
    else:
        block_type_list = [6, 5, 6]
        if topo[1]:
            block_type_list.append(5)
        if topo[2]:
            block_type_list.append(5)
        if topo[3]:
            block_type_list.append(5)
    block_type_list = np.concatenate((np.array(block_type_list), topo[4:]))
    return block_type_list


def simulate(x, bounds):
    ret = np.zeros((x.shape[0], 5))
    mean = bounds[0]
    delta = bounds[1] - bounds[0]
    x = x * delta + mean
    circuit_dir = './circuit_behavior'

    for aa in range(np.shape(x)[0]):
        sizing = x[aa]
        conf_file = os.path.join(circuit_dir, 'conf')
        param_file = os.path.join(circuit_dir, 'param')
        performance_file = os.path.join(circuit_dir, '3stage.ma0')
        sp_file = os.path.join(circuit_dir, '3stage.sp')
        output_name = os.path.join(circuit_dir, '3stage')
        name = []

        with open(conf_file, 'r') as f1:
            lines = f1.readlines()
            for l in lines:
                l = l.strip().split(' ')
                if l[0] == 'des_var':
                    name.append(l[1])

        power = 0

        with open(param_file, 'w') as f:
            for i in range(len(name)):
                if name[i][0] in ['g', 'G']:
                    power += np.abs(sizing[i])
                f.write('.param ' + name[i] + ' = ' + str(sizing[i]) + '\n')

        power = power * 1.8 / 20

        # hspice simulation
        os.system('hspice64 -C -i {} -o {}'.format(sp_file, output_name))

        # get results
        with open(performance_file, 'r') as f:
            lines = f.readlines()
            line = lines[4].strip().split()
            gain = convert(line[0])

            if convert(line[0]) < 0 or line[1] == "failed":
                ugf = 0
                pm = 0
            else:
                ugf = convert(line[1])
                ph = float(line[2])

                result_file = os.path.join(circuit_dir, '3stage.lis')
                id = []
                phase = []
                with open(result_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        line = line.strip().split()
                        if len(line) > 0 and line[0] == 'freq':
                            id.append(i)
                        if len(id) == 2:
                            break
                    for i in range(901):
                        if convert(lines[id[0] + i + 2].strip().split()[1]) < 0:
                            break
                    for j in range(i + 1):
                        phase.append(convert(lines[id[1] + j + 2].strip().split()[1]))
                valid = True
                for i in range(1, len(phase)):
                    if phase[i] > phase[i - 1]:
                        valid = False
                        break
                if not valid:
                    pm = -180
                elif np.abs(phase[0]) > 90:
                    pm = ph
                else:
                    pm = ph + 180

        y = np.zeros(5)
        y[0] = - ugf / power / 1e11
        y[1] = - (gain - 85) / 85 #spec1
        y[2] = - (ugf - 7e5) / 7e5 #spec1
        y[3] = - (pm - 55) / 55 #spec1
        y[4] = (power - 2.5e-4) / 2.5e-4 #spec1

        #y[1] = - (gain - 80) / 80 #spec2
        #y[2] = - (ugf - 5e7) / 5e7 #spec2
        #y[3] = - (pm - 55) / 55 #spec2
        #y[4] = (power - 5e-3) / 5e-3 #spec2
        ret[aa] = y
    return ret
