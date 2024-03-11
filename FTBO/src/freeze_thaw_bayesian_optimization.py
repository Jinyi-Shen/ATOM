import numpy as np
from copy import deepcopy
import logging
import os
import pickle
from .init_random_uniform_single import init_random_uniform
from .base_task import BaseTask
from .freeze_thaw_model import FreezeThawGP
from .ei import EI
from .information_gain_mc_freeze import InformationGainMC
from scipy.stats import norm
from .GP import GP
from .BO import BO
from .util import *
from scipy.optimize import fmin_l_bfgs_b
import traceback
import scipy.io as sio
import time

logger = logging.getLogger(__name__)
logging.basicConfig(filename='tmp.log', level=logging.INFO)

VU_PRINT = 2
SAVE_FILE = 'freezeData.pkl'


#	0 = minimal
# 	1 = only details
# 	2 = even more details


class FreezeThawBO():

    def __init__(self,
                 task,
                 init_points=5,
                 basket_size=10,
                 basketOld_X=None,
                 basketOld_Y=None,
                 threshold=0,
                 n_bfgs=10,
                 spec=1):
        """
        Class for the Freeze Thaw Bayesian Optimization by Swersky et al.
        
        Parameters
        ----------
        basketOld_X: ndarray(N,D)
            Basket with the old configurations, with N as the number of configurations
            and D as the number of dimensions
        basketOld_Y: ndarray(N,S)
            Basket with the learning curves of the old configurations, with N as the number
            of configurations and S as the number of steps
        task: TaskObject
            Task object that contains the objective function and additional
            meta information such as the lower and upper bound of the search
            space.
        """

        self.X = None
        self.Y = None
        self.ys = None

        self.freezeModel = FreezeThawGP(lg=False)

        self.task = task
        self.dim = self.task.X_lower.shape[0]
        self.best_constr = np.inf
        self.best_y = np.inf
        self.best_x = np.zeros((self.dim))
        self.n_bfgs = n_bfgs

        self.model_untrained = True

        self.init_points = init_points
        self.basket_size = basket_size

        self.basketOld_X = basketOld_X
        self.basketOld_Y = basketOld_Y
        self.basketOld_topo = None
        self.topo_set = set()
        self.basket_indices = list()
        self.directory = "temp_configs_" + self.task.base_name
        self.time_dict = {}
        self.time_dict['model_time'] = []
        self.time_dict['opt_time'] = []
        self.time_dict['es_time'] = []

        self.num_cons = spec - 1

        self.all_configs = dict()
        self.total_num_confs = 0

        self.freezethaw_flag = 0
        self.debug = False
        self.threshold = threshold  # only points better than the threshold can be added to the basket

    def printBasket(self):
        print("::::																	::::")
        print("::::::::					The Current Basket of the BO: 					::::")
        for cId, cand in enumerate(self.basketOld_topo):
            print("::::::::		Candidate #", cId, ": ", cand, "\n:::::::: 					   	value = ",
                  self.basketOld_Y[cId])
        print("::::::::																	::::")

    def run(self, num_iterations=10):
        """
        Bayesian Optimization iterations
        
        Parameters
        ----------
        num_iterations: int
            How many times the BO loop has to be executed
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
            Function values of the already evaluated points
        
        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        flow_begin_i = time.time()
        best_y = np.inf
        best_cons = np.inf

        num_valid_init = 0

        self.basketOld_X = np.zeros((self.basket_size, self.dim))
        self.basketOld_Y = np.zeros(self.basket_size, dtype=object)
        self.basketOld_topo = np.zeros(self.basket_size, dtype=object)

        x0 = np.zeros((self.init_points, self.dim))
        ys = np.zeros(self.init_points, dtype=object)
        cons = np.zeros((self.init_points, self.num_cons))
        i = 0
        n_sim = 0
        while i < self.init_points:
            conf_now = init_random_uniform(self.task.X_lower, self.task.X_upper)
            logger.info("Evaluate: %s" % conf_now)
            topo = self.task.decode_topo(x=conf_now)
            if topo in self.topo_set:
                continue
            else:
                self.topo_set.add(topo)

                Results = self.task.objective_function()

                val_losses = Results[0]
                tmp_cons = Results[1:]
                # print('temp_cons', tmp_cons)

                cons1 = np.sum(np.maximum(tmp_cons, 0))
                if best_cons > 0:
                    if cons1 < best_cons:
                        best_y = val_losses - 10 * cons1
                        best_cons = cons1
                        # best_topo = topo
                else:
                    if cons1 == 0 and val_losses < best_y:
                        best_y = val_losses
                        # best_topo = topo

                if VU_PRINT >= 0:
                    print("In initing points, val_losses = ", val_losses)
                    print("In initing points, tmp_cons = ", tmp_cons)

                # storing configuration, learning curve, constr, activity, index in the basketOld
                ys[i] = np.asarray([val_losses])
                cons[i] = tmp_cons
                x0[i] = conf_now

                if val_losses < self.threshold:
                    if num_valid_init < self.basket_size:
                        self.all_configs[self.total_num_confs] = [conf_now, topo, ys[i], tmp_cons, True, num_valid_init]
                        self.basketOld_topo[num_valid_init] = topo
                        self.basket_indices.append(self.total_num_confs)
                        self.basketOld_Y[num_valid_init] = ys[i]
                        self.basketOld_X[num_valid_init, :] = deepcopy(conf_now)
                        num_valid_init += 1
                    else:
                        yy = []
                        for iii in range(len(self.basketOld_Y)):
                            yy.append(self.basketOld_Y[iii][0])
                        replace = np.argmax(np.array(yy))
                        self.basketOld_topo[replace] = topo
                        self.basket_indices[replace] = self.total_num_confs
                        self.basketOld_X[replace] = conf_now
                        self.basketOld_Y[replace] = ys[i]
                        self.all_configs[self.total_num_confs] = [conf_now, topo, ys[i], tmp_cons, True, replace]
                        self.all_configs[self.basket_indices[replace]][4] = False
                        self.all_configs[self.basket_indices[replace]][5] = -1
                else:
                    self.all_configs[self.total_num_confs] = [conf_now, topo, ys[i], tmp_cons, False, -1]
                n_sim += 15
                i += 1
                self.total_num_confs += 1

        Y = np.zeros((len(ys), 1))
        print('#########################################ys1##################################################', ys)
        for i in range(Y.shape[0]):
            Y[i, 0] = ys[i][-1]

        if VU_PRINT >= 1:
            print("ys after getting init points: ", ys)
            print("Y after gettting init points: ", Y)

        self.X = deepcopy(x0)
        self.ys = deepcopy(ys)
        self.freezeModel.X = self.freezeModel.x_train = self.X
        self.freezeModel.ys = self.freezeModel.y_train = self.ys

        self.cons = deepcopy(cons)

        self.freezeModel.Y = Y
        self.freezeModel.actualize()

        res_list = []
        logAllIters = []
        for ii in range(self.init_points, num_iterations):

            logger.info("Start iteration %d ... ", ii)
            print('######################iteration num: {:d} #################################'.format(ii))

            if VU_PRINT >= 1:
                print("freezeModel.X.shape is :", self.freezeModel.X.shape)

            if num_valid_init >= self.basket_size:
                self.freezethaw_flag = 1

            ####### Opt ################################################
            ### Model Training
            time_start = time.time()
            self.freezeModel.train(self.X, self.ys, do_optimize=True)
            self.freezeModel.actualize()
            dataset = {}
            dataset['train_x'] = self.X
            self.Cons_models = []
            for i in range(self.num_cons):
                dataset['train_y'] = self.cons[:, i][:, None]
                self.Cons_models.append(GP(dataset))
                self.Cons_models[i].train()

            print('Cons GP model constructing finished.')
            time_end = time.time()
            self.time_dict['model_time'].append(time_end - time_start)
            self.Y = getY(self.ys)
            self.get_best_y(self.X, self.Y, self.cons)

            ### choose the next candidates
            allx0 = np.zeros((self.n_bfgs, self.dim))
            for i in range(self.n_bfgs):
                x0 = init_random_uniform(self.task.X_lower, self.task.X_upper)
                x0 = self.optimize_wEI(x0)
                allx0[i] = x0.reshape(-1)

            wEI_tmp = self.calc_log_wEI_approx(allx0)
            idx = np.argsort(wEI_tmp)
            topo = ''
            for j in range(self.n_bfgs):
                topo = self.task.decode_topo(allx0[idx[self.n_bfgs - 1 - j]])
                if topo not in self.topo_set:
                    break
            assert topo != '', 'no new topo sampled'
            new_x = allx0[idx[self.n_bfgs - 1 - j]]
            time_end1 = time.time()
            self.time_dict['opt_time'].append(time_end1 - time_end)

            if not self.freezethaw_flag:
                self.time_dict['es_time'].append(0)
                thisIter = {}
                thisIter["config_id"] = self.total_num_confs
                thisIter["old"] = False
                logger.info("Evaluate candidate %s" % (str(new_x)))

                Results = self.task.objective_function()
                self.topo_set.add(topo)
                val_losses = Results[0]
                tmp_cons = Results[1:]
                print('val_losses and tmp_cons for new_x', val_losses, tmp_cons)

                logger.info("Configuration achieved a performance of %f " % (val_losses))
                # new
                self.X = np.append(self.X, new_x[np.newaxis, :], axis=0)
                newArray = np.zeros(1, dtype=object)
                newArray[0] = np.asarray([val_losses])

                self.ys = np.append(self.ys, newArray, axis=0)
                self.cons = np.append(self.cons, tmp_cons[None, :], axis=0)

                # add data to basket
                if val_losses < self.threshold:
                    self.all_configs[self.total_num_confs] = [new_x, topo, self.ys[-1], tmp_cons, True, num_valid_init]
                    self.basketOld_topo[num_valid_init] = topo
                    self.basket_indices.append(self.total_num_confs)
                    self.basketOld_Y[num_valid_init] = self.ys[-1]
                    self.basketOld_X[num_valid_init, :] = deepcopy(new_x).reshape(-1)
                    num_valid_init += 1
                else:
                    self.all_configs[self.total_num_confs] = [new_x, topo, self.ys[-1], tmp_cons, False, -1]

                if VU_PRINT >= 1:
                    print("After a new model added, X = ", self.X)
                    print("After a new model added, ys = ", self.ys)
                    print("After a new model added, cons = ", self.cons)

                if VU_PRINT >= 1:
                    print("In freeze_thaw_bo, run, after setting all_configs: ")
                    print("all_configs = ", self.all_configs)
                    print("total_num_confs = ", self.total_num_confs, "length of all_configs = ",
                          len(self.all_configs))

                self.total_num_confs += 1
                n_sim += 15

            else:
                time_start_es = time.time()
                ig = InformationGainMC(model=self.freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper)
                ig.update(self.freezeModel, calc_repr=True)

                H = ig.compute()
                zb = deepcopy(ig.zb)
                lmb = deepcopy(ig.lmb)

                print('H: {}'.format(H))
                # Fantasize over the old and the new configurations
                num_old = self.basket_size
                fant_old = np.zeros(num_old)

                for i in range(num_old):
                    conf_index = self.basket_indices[i]
                    fv = self.freezeModel.predict(option='old', conf_num=conf_index)
                    fant_old[i] = fv[0]

                if VU_PRINT >= 0:
                    print("So fantasized loss of old models: ", fant_old)

                num_new = 1
                fant_new, _ = self.freezeModel.predict(xprime=new_x, option='new')

                if VU_PRINT >= 0:
                    print("So fantasized loss of new models: ", fant_new)

                Hfant = np.zeros(num_old + 1)

                for i in range(num_old):
                    conf_index = self.basket_indices[i]
                    no_improvement = 5
                    if self.ys[conf_index].shape[0] >= no_improvement and np.max(
                            np.abs(self.ys[conf_index][-no_improvement:-1] - self.ys[conf_index][-1])) < 1e-6:
                        Hfant[i] = H + 1000
                    else:
                        freezeModel = deepcopy(self.freezeModel)
                        y_i = freezeModel.ys[conf_index]
                        y_i = np.append(y_i, np.array([fant_old[i]]), axis=0)
                        freezeModel.ys[conf_index] = y_i
                        freezeModel.train(X=freezeModel.X, Y=freezeModel.ys, do_optimize=False)

                        ig1 = InformationGainMC(model=freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper,
                                                sampling_acquisition=EI)
                        ig1.actualize(zb, lmb)
                        ig1.update(freezeModel)
                        Hfant[i] = ig1.compute()

                freezeModel = deepcopy(self.freezeModel)

                newX = np.append(freezeModel.X, new_x[np.newaxis, :], axis=0)
                ysNew = np.zeros(len(freezeModel.ys) + 1, dtype=object)
                for i in range(len(freezeModel.ys)):
                    ysNew[i] = freezeModel.ys[i]

                ysNew[-1] = np.array([fant_new])
                freezeModel.train(X=newX, Y=ysNew, do_optimize=False)

                ig1 = InformationGainMC(model=freezeModel, X_lower=self.task.X_lower,
                                        X_upper=self.task.X_upper,
                                        sampling_acquisition=EI)
                ig1.actualize(zb, lmb)
                ig1.update(freezeModel)
                H1 = ig1.compute()
                Hfant[-1] = H1

                # Comparison of the different values
                infoGain = -(Hfant - H)
                max_infoGain = np.max(infoGain)
                candidate_list = []
                candidate_basket_list = []
                for i in range(infoGain.shape[0]):
                    if np.abs(infoGain[i] - max_infoGain) < 1e-6:
                        candidate_list.append(i)
                        if i <= infoGain.shape[0] - 2:
                            candidate_basket_list.append(i)
                if len(candidate_list) == 1:
                    winner = candidate_list[0]
                else:
                    if len(candidate_basket_list) <= 1:
                        winner = candidate_basket_list[0]
                    else:
                        best_candidate = self.basketOld_X[candidate_basket_list]
                        wEI_tmp = self.calc_log_wEI_approx(best_candidate)
                        print('Hfant same value, compute wEI: {}'.format(wEI_tmp))
                        best_EI = np.argmax(wEI_tmp)
                        winner = candidate_list[best_EI]
                thisIter = {}

                print('the winner is index: {:d}'.format(winner))
                time_end_es = time.time()
                self.time_dict['es_time'].append(time_end_es - time_start_es)
                # run an old configuration and actualize basket
                if winner <= ((len(Hfant) - 1) - num_new):
                    print('###################### run old config ######################')

                    conf_to_run = self.basketOld_X[winner]

                    logger.info("Evaluate candidate %s" % (str(conf_to_run)))
                    topo = self.task.decode_topo(conf_to_run)
                    Results = self.task.objective_function()
                    val_losses = Results[0]
                    tmp_cons = Results[1:]

                    logger.info("Configuration achieved a performance of %f " % (val_losses))

                    self.basketOld_Y[winner] = np.append(self.basketOld_Y[winner], np.asarray([val_losses]))
                    index_now = self.basket_indices[winner]

                    thisIter["config_id"] = index_now
                    thisIter["old"] = True

                    self.all_configs[self.total_num_confs] = [conf_to_run, topo, self.basketOld_Y[winner], tmp_cons,
                                                              True, winner]
                    self.ys[index_now] = self.basketOld_Y[winner]  # new
                    ### update constraint
                    self.cons[index_now] = tmp_cons
                    n_sim += 10
                    # else run the new proposed configuration and actualize
                else:
                    print('###################### run new config #' + str(
                        self.total_num_confs) + ' ######################')
                    thisIter["config_id"] = self.total_num_confs
                    thisIter["old"] = False

                    # winner = winner - num_old
                    winner = 0  # num_new == 1

                    logger.info("Evaluate candidate %s" % (str(new_x)))

                    Results = self.task.objective_function()
                    self.topo_set.add(topo)
                    val_losses = Results[0]
                    tmp_cons = Results[1:]

                    logger.info("Configuration achieved a performance of %f " % (val_losses))

                    newArray = np.zeros(1, dtype=object)
                    newArray[0] = np.asarray([val_losses])
                    self.X = np.append(self.X, new_x[None, :], axis=0)
                    self.ys = np.append(self.ys, newArray, axis=0)
                    self.cons = np.append(self.cons, tmp_cons[None, :], axis=0)

                    # add data to basket
                    if val_losses < self.threshold:
                        if np.min(np.abs(infoGain + 1000)) < 1e-6:
                            replace = np.argmin(np.abs(infoGain + 1000))
                            self.basketOld_topo[replace] = topo
                        else:
                            replace = self.get_min_ei(self.basketOld_X, self.basketOld_Y)
                            self.basketOld_topo[replace] = topo
                        self.basket_indices[replace] = self.ys.shape[0] - 1
                        self.basketOld_X[replace] = new_x
                        self.basketOld_Y[replace] = self.ys[-1]
                        self.all_configs[self.total_num_confs] = [new_x, topo, self.ys[-1], tmp_cons, True, replace]
                        self.all_configs[self.basket_indices[replace]][4] = False
                        self.all_configs[self.basket_indices[replace]][5] = -1
                    else:
                        self.all_configs[self.total_num_confs] = [new_x, topo, self.ys[-1], tmp_cons, False, -1]
                    n_sim += 15

                    if VU_PRINT >= 1:
                        print("After a new model added, X = ", self.X)
                        print("After a new model added, ys = ", self.ys)
                        print("After a new model added, cons = ", self.cons)

                    if VU_PRINT >= 1:
                        print("In freeze_thaw_bo, run, after setting all_configs: ")
                        print("all_configs = ", self.all_configs)
                        print("total_num_confs = ", self.total_num_confs, "length of all_configs = ",
                              len(self.all_configs), " winner = ", winner, " length of basketOld_Y = ",
                              len(self.basketOld_Y))

                self.total_num_confs += 1

            logAllIters.append(thisIter)
            pickle.dump(logAllIters, open(SAVE_FILE, 'wb'))

            if VU_PRINT >= 1:
                self.printBasket()

            cons1 = np.sum(np.maximum(tmp_cons, 0))
            if best_cons > 0:
                if cons1 < best_cons:
                    best_y = val_losses - 10 * cons1
                    best_cons = cons1
            else:
                if cons1 == 0 and val_losses < best_y:
                    best_y = val_losses
            flow_end_i = time.time()
            res = [val_losses - 10 * cons1, cons1, best_y, best_cons, n_sim, flow_end_i - flow_begin_i]
            res_list.append(res)

            print("::::::::::::::::::::::   CURRENT INCUMBENT: ", self.best_x, ", value = ", self.best_y)

            with open('all_configs.pkl', 'wb') as c:
                pickle.dump(self.all_configs, c)

        return self.best_x, self.best_y, self.time_dict, res_list

    def calc_EI(self, x):
        EI = np.ones(x.shape[0])
        if self.best_constr <= 0:
            py, ps2 = self.freezeModel.predict(x)
            py = py.reshape(-1)
            ps2 = ps2.reshape(-1)
            ps = np.sqrt(ps2) + 0.000001
            normed = -(py - self.best_y) / ps
            EI = ps * (normed * normcdf(normed) + normpdf(normed))
        return EI

    def calc_PI(self, x):
        PI = np.ones(x.shape[0])
        for i in range(self.num_cons):
            py, ps2 = self.Cons_models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI * normcdf(-py / ps)
        return PI

    def calc_log_PI(self, x):
        PI = np.zeros(x.shape[0])
        for i in range(self.num_cons):
            py, ps2 = self.Cons_models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI + logphi_vector(-py / ps)
        return PI

    def calc_log_wEI(self, x):
        log_EI = np.log(np.maximum(0.000001, self.calc_EI(x)))
        log_PI = self.calc_log_PI(x)
        log_wEI = log_EI + log_PI
        return log_wEI

    def calc_log_wEI_approx(self, x):
        log_EI_approx = np.zeros(x.shape[0])
        if (self.best_constr <= 0):
            py, ps2 = self.freezeModel.predict(x)
            py = py.reshape(-1)
            ps2 = ps2.reshape(-1)
            ps = np.sqrt(ps2) + 0.000001
            normed = -(py - self.best_y) / ps
            EI = ps * (normed * normcdf(normed) + normpdf(normed))
            log_EI = np.log(np.maximum(0.000001, EI))

            tmp = np.minimum(-40, normed) ** 2
            log_EI_approx = np.log(ps) - tmp / 2 - np.log(tmp - 1)
            log_EI_approx = log_EI * (normed > -40) + log_EI_approx * (normed <= -40)

        log_PI = self.calc_log_PI(x)
        log_wEI_approx = log_EI_approx + log_PI
        return log_wEI_approx

    def get_min_ei(self, basketOld_X, basketOld_Y):
        Ylist = -1 * getY(basketOld_Y)
        Lb_list = np.zeros(len(basketOld_Y))
        for i in range(len(basketOld_Y)):
            Lb_list[i] = Ylist[i]

        minIndex = np.argmin(Lb_list)

        if VU_PRINT >= 1:
            print(
                "::::			Finding the min lower bound in the current basket to be replaced by the new config  		::::")
            print("::::			Lb_list is :", Lb_list)
            print(":::: winner = ", minIndex, " : ", Lb_list[minIndex])

        return minIndex

    def get_best_y(self, x, y, cons):
        for i in range(y.shape[0]):
            constr = np.maximum(cons[i, :], 0).sum()
            if constr < self.best_constr and self.best_constr > 0:
                self.best_constr = constr
                self.best_y = np.copy(y[i])
                self.best_x = np.copy(x[i, :])
        if self.best_constr <= 0:
            best = np.argmin(y)
            self.best_x = np.copy(x[best])
            self.best_y = np.copy(y[best])

    def optimize_constr(self, x):
        x0 = np.copy(x).reshape(-1)
        best_x = np.copy(x)
        best_loss = np.inf
        tmp_loss = np.inf

        def loss(x0):
            nonlocal tmp_loss
            x0 = x0.reshape(-1, self.dim)
            py, ps2 = self.freezeModel.predict(x0)
            py = py.reshape(-1)
            ps2 = ps2.reshape(-1)
            tmp_loss = py.sum()
            for i in range(self.num_cons):
                py, ps2 = self.Cons_models[i].predict(x0)
                tmp_loss += np.maximum(0, py).sum()
            return tmp_loss

        def callback(x):
            nonlocal best_x
            nonlocal best_loss
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                best_x = np.copy(x)

        try:
            fmin_l_bfgs_b(loss, x0, approx_grad=True, bounds=[[-1.5, 1.5]] * x.size, maxiter=1000, m=100, iprint=self.debug,
                          callback=callback)
        except np.linalg.LinAlgError:
            print('Optimizing constrains. Increase noise term and re-optimization')
            x0 = np.copy(best_x).reshape(-1)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(loss, x0, approx_grad=True, bounds=[[-1.5, 1.5]] * x.size, maxiter=1000, m=10,
                              iprint=self.debug, callback=callback)
            except:
                print('Optimizing constrains. Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())
        except:
            print('Optimizing constrains. Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())

        best_x = best_x.reshape(-1, self.dim)
        return best_x

    def optimize_wEI(self, x):
        x0 = np.copy(x).reshape(-1)
        best_x = np.copy(x)
        best_loss = np.inf
        tmp_loss = np.inf

        def loss(x0):
            nonlocal tmp_loss
            x0 = x0.reshape(-1, self.dim)
            tmp_loss = - self.calc_log_wEI_approx(x0)
            #            tmp_loss = - self.calc_wEI(x0)
            tmp_loss = tmp_loss.sum()
            return tmp_loss

        def callback(x):
            nonlocal best_x
            nonlocal best_loss
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                best_x = np.copy(x)

        try:
            fmin_l_bfgs_b(loss, x0, approx_grad=True, bounds=[[-1.5, 1.5]] * x.size, maxiter=1000, m=100, iprint=self.debug,
                          callback=callback)
        except np.linalg.LinAlgError:
            print('Optimizing constrains. Increase noise term and re-optimization')
            x0 = np.copy(best_x).reshape(-1)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(loss, x0, approx_grad=True, bounds=[[-1.5, 1.5]] * x.size, maxiter=1000, m=10,
                              iprint=self.debug, callback=callback)
            except:
                print('Optimizing acquisition function, Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())
        except:
            print('Optimizing acquisition function, Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())

        best_x = best_x.reshape(-1, self.dim)
        return best_x


def getY(ys):
    Y = np.zeros((len(ys), 1))
    for i in range(Y.shape[0]):
        Y[i, :] = ys[i][-1]
    return Y
