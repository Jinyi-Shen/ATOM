# import autograd.numpy as np
# from autograd import value_and_grad
from scipy.optimize import fmin_l_bfgs_b, minimize
import numpy as np
import traceback
import sys
from .GP import GP
from .util import *
import random
from autograd import value_and_grad


class BO:
    def __init__(self, dataset, bfgs_iter=300):
        self.train_x = np.copy(dataset['train_x'])
        self.train_y = np.copy(dataset['train_y'])
        self.dim = self.train_x.shape[1]
        self.outdim = self.train_y.shape[1]
        self.num_train = self.train_y.shape[0]
        self.bfgs_iter = bfgs_iter
        self.construct_model()

        self.best_constr = np.inf
        self.best_y = np.zeros((self.outdim))
        self.best_y[0] = np.inf
        self.best_x = np.zeros((self.dim))
        self.get_best_y(self.train_x, self.train_y)


    def construct_model(self):
        dataset = {}
        dataset['train_x'] = self.train_x
        self.models = []
        for i in range(self.outdim):
            dataset['train_y'] = self.train_y[:, i][:, None]
            self.models.append(GP(dataset, bfgs_iter=self.bfgs_iter))
            self.models[i].train()
        # print('BO. GP model constructing finished.')

    def get_best_y(self, x, y):
        for i in range(y.shape[0]):
            constr = np.maximum(y[i, 1:], 0).sum()
            if constr < self.best_constr and self.best_constr > 0:
                self.best_constr = constr
                self.best_y = np.copy(y[i, :])
                self.best_x = np.copy(x[i, :])
            elif constr <= 0 and self.best_constr <= 0 and y[i, 0] < self.best_y[0]:
                self.best_constr = constr
                self.best_y = np.copy(y[i, :])
                self.best_x = np.copy(x[i, :])

    def rand_x(self, n=1):
        x = np.random.uniform(0, 1, (n, self.dim))
        #        ''' improve the possibility for random sampling points x near the current best x'''
        #        tmp = np.random.uniform(0,1,(n))
        #        idx = (tmp < 0.2)
        #        x[:,idx] = (0.1*np.random.uniform(-0.5,0.5,(self.dim,idx.sum())).T + self.best_x).T
        #        x[:,idx] = np.maximum(-0.5, np.minimum(0.5, x[:,idx]))
        return x

    def calc_EI(self, x):
        EI = np.ones(x.shape[0])
        if self.best_constr <= 0:
            py, ps2 = self.models[0].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            normed = -(py - self.best_y[0]) / ps
            EI = ps * (normed * normcdf(normed) + normpdf(normed))
        return EI

    def calc_PI(self, x):
        PI = np.ones(x.shape[0])
        for i in range(1, self.outdim):
            py, ps2 = self.models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI * normcdf(-py / ps)
        return PI

    def calc_log_PI(self, x):
        PI = np.zeros(x.shape[0])
        for i in range(1, self.outdim):
            py, ps2 = self.models[i].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            PI = PI + logphi_vector(-py / ps)
        return PI

    def calc_wEI(self, x):
        EI = self.calc_EI(x)
        PI = self.calc_PI(x)
        wEI = EI * PI
        return wEI

    def calc_log_wEI(self, x):
        log_EI = np.log(np.maximum(0.000001, self.calc_EI(x)))
        log_PI = self.calc_log_PI(x)
        log_wEI = log_EI + log_PI
        return log_wEI

    def calc_log_wEI_approx(self, x):
        log_EI_approx = np.zeros(x.shape[0])
        if (self.best_constr <= 0):
            py, ps2 = self.models[0].predict(x)
            ps = np.sqrt(ps2) + 0.000001
            normed = -(py - self.best_y[0]) / ps
            EI = ps * (normed * normcdf(normed) + normpdf(normed))
            log_EI = np.log(np.maximum(0.000001, EI))

            tmp = np.minimum(-40, normed) ** 2
            log_EI_approx = np.log(ps) - tmp / 2 - np.log(tmp - 1)
            log_EI_approx = log_EI * (normed > -40) + log_EI_approx * (normed <= -40)

        log_PI = self.calc_log_PI(x)
        log_wEI_approx = log_EI_approx + log_PI
        return log_wEI_approx

    def optimize_tt_constr(self, x):
        x0 = np.copy(x).reshape(-1)
        best_x = np.copy(x)
        best_loss = np.inf
        tmp_loss = np.inf
        bounds = [[0, 1]] * x.size

        def loss(x0):
            nonlocal tmp_loss
            x0 = x0.reshape(-1, self.dim)
            py, ps2 = self.models[0].predict(x0)
            tmp_loss = py.sum()
            for i in range(1, self.outdim):
                py, ps2 = self.models[i].predict(x0)
                tmp_loss += np.maximum(0, py).sum()
            return tmp_loss

        def callback(x):
            nonlocal best_x
            nonlocal best_loss
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                best_x = np.copy(x)

        # gloss = value_and_grad(loss)
        '''
        try:
            #starting point for fmin_l_bfgs_b should be a one-dimension vector
            #fmin_l_bfgs_b(gloss, x0, bounds=[[0,1]]*x.size, maxiter=2000, m=100, iprint=False, callback=callback)
            #fmin_l_bfgs_b(loss, x0, approx_grad=True, bounds=[[0,1]]*x.size, maxiter=2000, m=100, iprint=True, callback=callback)
            res = minimize(loss, x0, bounds=[[0,1]]*x.size, method='SLSQP',callback=callback)
        #except np.linalg.LinAlgError:
        #    print('Optimizing constrains. Increase noise term and re-optimization')
        #    x0 = np.copy(best_x).reshape(-1)
        #    x0[0] += 0.01
        #    try:
        #        #fmin_l_bfgs_b(gloss, x0, bounds=[[0,1]]*x.size, maxiter=2000, m=10, iprint=False, callback=callback)
        #        fmin_l_bfgs_b(loss, x0, approx_grad=True, bounds=[[0, 1]]*x.size, maxiter=2000, m=10, iprint=False, callback=callback)
        #    except:
        #        print('Optimizing constrains. Exception caught, L-BFGS early stopping...')
        #        print(traceback.format_exc())
        except:
            pass
            # print('Optimizing constrains. Exception caught, L-BFGS early stopping...')
            # print(traceback.format_exc())
        '''
        gloss = value_and_grad(loss)
        try:
            fmin_l_bfgs_b(gloss, x0, bounds=bounds, maxiter=2000, m=100, iprint=self.debug,
                          callback=callback)
        except np.linalg.LinAlgError:
            # print('Optimizing constrains. Increase noise term and re-optimization')
            x0 = np.copy(best_x).reshape(-1)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(gloss, x0, bounds=bounds, maxiter=2000, m=10, iprint=self.debug,
                              callback=callback)
            except:
                pass
                # print('Optimizing constrains. Exception caught, L-BFGS early stopping...')
                # print(traceback.format_exc())
        except:
            pass
            # print('Optimizing constrains. Exception caught, L-BFGS early stopping...')
            # print(traceback.format_exc())

        best_x = best_x.reshape(-1, self.dim)
        # best_x = res.x.reshape(-1, self.dim)
        return best_x

    def optimize_tt_wEI(self, x):
        x0 = np.copy(x).reshape(-1)
        best_x = np.copy(x)
        best_loss = np.inf
        tmp_loss = np.inf

        def loss(x0):
            nonlocal tmp_loss
            x0 = x0.reshape(-1, self.dim)
            tmp_loss = - self.calc_log_wEI_approx(x0)
            tmp_loss = tmp_loss.sum()
            return tmp_loss

        def callback(x):
            nonlocal best_x
            nonlocal best_loss
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                best_x = np.copy(x)

        gloss = value_and_grad(loss)

        try:
            # fmin_l_bfgs_b(gloss, x0, bounds=[[0,1]]*x.size, maxiter=2000, m=100, iprint=False, callback=callback)
            fmin_l_bfgs_b(gloss, x0, bounds=[[0, 1]] * x.size, maxiter=2000, m=100, iprint=False,
                          callback=callback)
        except np.linalg.LinAlgError:
            # print('Acquisition func optimization error, Increase noise term and re-optimization')
            x0 = np.copy(best_x).reshape(-1)
            x0[0] += 0.01
            try:
                # fmin_l_bfgs_b(gloss, x0, bounds=[[0,1]]*x.size, maxiter=2000, m=10, iprint=False, callback=callback)
                fmin_l_bfgs_b(gloss, x0, bounds=[[0, 1]] * x.size, maxiter=2000, m=10, iprint=False,
                              callback=callback)
            except:
                pass
                # print('Optimizing acquisition function, Exception caught, L-BFGS early stopping...')
                # print(traceback.format_exc())
        except:
            pass
            # print('Optimizing acquisition function, Exception caught, L-BFGS early stopping...')
            # print(traceback.format_exc())

        best_x = best_x.reshape(-1, self.dim)
        return best_x

    def predict(self, test_x):
        num_test = test_x.shape[0]
        py = np.zeros((num_test, self.outdim))
        ps2 = np.zeros((num_test, self.outdim))
        for i in range(self.outdim):
            py[i], ps2[i] = self.models[i].predict(test_x)
        return py, ps2
