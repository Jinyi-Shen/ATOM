# encoding=utf8
import scipy
import numpy as np
import logging
import traceback
from .base_model import BaseModel
from numpy.linalg import inv
from numpy.linalg import solve
from scipy.linalg import block_diag
from .cmaes import CMAES
from scipy.optimize import fmin_l_bfgs_b
from .base_prior import BasePrior, TophatPrior, \
LognormalPrior, HorseshoePrior, UniformPrior

logger = logging.getLogger(__name__)

VU_PRINT = 0

class FreezeThawGP(BaseModel):

    def __init__(self,
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 invChol=True,
                 horse=True, 
                 samenoise=True,
                 lg=True):
        """
        Parameters
        ----------
        x_train: ndarray(N,D)
            The input training data for all GPs
        y_train: ndarray(N,T)
            The target training data for all GPs. The ndarray can be of dtype=object,
            if the curves have different lengths
        x_test: ndarray(*,D)
            The current test data for the GPs, where * is the number of test points
        sampleSet : ndarray(S,H)
            Set of all GP hyperparameter samples (S, H), with S as the number of samples and H the number of
            GP hyperparameters. This option should only be used, if the GP-hyperparameter samples already exist
        """

        self.X = x_train
        self.ys = y_train
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.Y = None
        
        self.invChol=invChol
        self.horse = horse
        self.samenoise = samenoise
        
        self.uPrior = UniformPrior(minv=0, maxv=10)
        self.lnPrior = LognormalPrior(sigma=0.1, mean=0.0)
        self.hPrior = HorseshoePrior()

        self.bfgs_iter = 2000
        self.hyperbounds = [[-4,4]] * 10 #hyperbounds
        if not self.samenoise:
            self.hyperbounds = self.hyperbounds + [[-4,4], [-3,3], [-4,4], [-10,-1], [-10,-1]]
        else:
            self.hyperbounds = self.hyperbounds + [[-4,4], [-3,3], [-4,4], [-10,-1]]

        if x_train is not None:
            self.activated = False

        self.lg = lg
        super(FreezeThawGP, self).__init__()

    def actualize(self):
        self.C = np.zeros((self.x_train.shape[0], self.x_train.shape[0]))
        self.mu = np.zeros((self.x_train.shape[0], 1))
        self.activated = False

    # Initialize hyper_parameters
    def get_default_theta(self):
        # number of length scales
        flex = self.x_train.shape[-1]

        if not self.samenoise:
            #theta0, alpha, beta, noiseHyper, noiseCurve
            fix = 5
        else:
            #theta0, alpha, beta, noise
            fix = 4

        # sample length scales for GP over configs
        p0a = self.uPrior.sample_from_prior(n_samples=(1, flex))

        # sample amplitude for GP over configs and alpha and beta for GP over curve
        p0b = self.lnPrior.sample_from_prior(n_samples=(1, 3))

        p0 = np.append(p0a, p0b, axis=1)

        #hPrior = HorseshoePrior()
        if not self.samenoise:
            if not self.horse:
                p0d = self.lnPrior.sample_from_prior(n_samples=(1, 2))
            else:
                p0d = np.abs(self.hPrior.sample_from_prior(
                    n_samples=(1, 2)))
        else:
            if not self.horse:
                p0d = self.lnPrior.sample_from_prior(n_samples=(1, 1))
            else:
                p0d = np.abs(self.hPrior.sample_from_prior(
                    n_samples=(1, 1)))

        p0 = np.append(p0, p0d, axis=1)

        return p0

    
    
    def train(self, X=None, Y=None, do_optimize=True):
        """
        Estimates the GP hyperparameter by maximizing the marginal
        loglikelihood   
        
        Parameters
        ----------
        x_train: ndarray(N,D)
            The input training data for all GPs
        y_train: ndarray(T,N)
            The target training data for all GPs. The ndarray can be of dtype=object,
            if the curves have different lengths.   
        """

        if X is not None:
            self.X = X
            self.x_train = X
        if Y is not None:
            self.ys = Y
            self.y_train = Y
            self.Y = np.zeros((len(Y), 1))
            for i in range(self.Y.shape[0]):
                self.Y[i, :] = Y[i][-1]

        self.m_const = self.get_mconst()

        if do_optimize:
            self.actualize()
            theta0 = self.get_default_theta()
            theta0 = theta0.reshape(-1)
            
            self.loss = np.inf
            self.theta = np.copy(theta0)

            nlz = -1 * self.marginal_likelihood(theta0)

            def loss(theta):
                nlz = -1 * self.marginal_likelihood(theta)
                return nlz

            def callback(theta):
                if self.nlz < self.loss:
                    self.loss = self.nlz
                    self.theta = np.copy(theta)
            try:
                fmin_l_bfgs_b(loss, theta0, approx_grad=True, bounds=self.hyperbounds, maxiter=self.bfgs_iter, m = 100, iprint=False, callback=callback)
            except np.linalg.LinAlgError:
                print('GP. Increase noise term and re-optimization')
                theta0 = np.copy(self.theta)
                theta0[-1] += np.log(10)
                try:
                    fmin_l_bfgs_b(loss, theta0, approx_grad=True, bounds=self.hyperbounds, maxiter=self.bfgs_iter, m=10, iprint=False, callback=callback)
                except:
                    print('GP. Exception caught, L-BFGS early stopping...')
            except:
                print('GP. Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())


            print('self.theta after train', self.theta)

    
    def predict(self, xprime=None, option='asympt', conf_num=0, from_step=None, further_steps=1, full_cov=False):
        """
        Predict using one of thre options: (1) predicion of the asymtote given a new configuration,
        (2) prediction of a new step of an old configuration, (3) prediction of steps of a curve of 
        a completely new configuration

        Parameters
        ----------
        xprime: ndarray(N,D)
            The new configuration(s)
        option: string
            The prediction type: 'asympt', 'old', 'new'
        conf_num: integer
            The index of an old configuration of which a new step is predicted
        from_step: integer
            The step from which the prediction begins for an old configuration.
            If none is given, it is assumend one is predicting from the last step
        further_steps: integer
            How many steps must be predicted from 'from_step'/last step onwards

        Results
        -------
        return: ndarray(N, steps), ndarray(N, steps)
            Mean and variance of the predictions
        """
        if option == 'asympt':
            if not full_cov:
                mu, std2 = self.pred_asympt(xprime)
            else:
                mu, std2, cov = self.pred_asympt(xprime, full_cov=full_cov)
        elif option == 'old': 
            if from_step is None:
                mu, std2 = self.pred_old(
                    conf_num=conf_num + 1, steps=further_steps)
            else:
                mu, std2 = self.pred_old(
                    conf_num=conf_num + 1, steps=further_steps, fro=from_step)
        elif option == 'new':
            mu, std2 = self.pred_new(
                steps=further_steps, xprime=xprime, asy=False)

        if type(mu) != np.ndarray:
            mu = np.array([[mu]])
        elif len(mu.shape)==1:
            mu = mu[:,None]

        if not full_cov:
            return mu, std2
        else:
            return mu, cov


    def marginal_likelihood(self, theta):
        """
        Calculates the marginal_likelikood for both the GP over hyperparameters and the GP over the training curves

        Parameters
        ----------
        theta: all GP hyperparameters

        Results
        -------
        marginal likelihood: float
            the resulting marginal likelihood
        """

        x = self.x_train
        y = self.y_train

        flex = x.shape[-1]
        theta_d = np.exp(theta[:flex])
        if not self.samenoise:
            theta0, alpha, beta, noiseHyper, noiseCurve = np.exp(theta[flex:])
        else:
            theta0, alpha, beta, noise = np.exp(theta[flex:])
            noiseHyper = noiseCurve = noise

        self.m_const = self.get_mconst()

        y_vec = self.getYvector(y)
        self.y_vec = y_vec
        
        O = self.getOmicron(y)

        kx = self.kernel_hyper(x, x, theta)
        if kx is None:
            return -np.inf

        if self.lg:
            Lambda, gamma = self.lambdaGamma(self.m_const, theta)
        else:
            Lambda, gamma = self.gammaLambda(self.m_const, theta)
        if Lambda is None or gamma is None:
            return -np.inf

        kx_inv = self.invers(kx)
        if kx_inv is None:
            return -np.inf

        kx_inv_plus_L = kx_inv + Lambda

        kx_inv_plus_L_inv = self.invers(kx_inv_plus_L)
        if kx_inv_plus_L_inv is None:
            return -np.inf

        kt = self.getKt(y, theta)
        if kt is None:
            return -np.inf

        kt_inv = self.invers(kt)
        if kt_inv is None:
            return -np.inf

        y_minus_Om = y_vec - np.dot(O, self.m_const)

        logP = -(1 / 2.) * np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)) + (1 / 2.) * np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))\
               - (1 / 2.) * (self.nplog(np.linalg.det(kx_inv_plus_L)) + self.nplog(np.linalg.det(kx)
                                                                                   ) + self.nplog(np.linalg.det(kt)))
        lp = logP[0,0]
        self.nlz = -1 * lp

        return lp

    def get_mconst(self):
        m_const = np.zeros((len(self.y_train), 1))
        
        for i in range(self.y_train.shape[0]):
            mean_i = np.mean(self.y_train[i], axis=0)
            m_const[i, :] = mean_i

        return m_const

    def pred_asympt(self, xprime, full_cov=False, show=False):
        """
        Given new configuration xprime, it predicts the probability distribution of
        the new asymptotic mean, with mean and covariance of the distribution
        
        Parameters
        ----------
        xprime: ndarray(number_configurations, D)
            The new configurations, of which the mean and the std2 are being predicted
        
        Returns
        -------
        mean: ndarray(len(xprime))
            predicted means for each one of the test configurations
        std2: ndarray(len(xprime))
            predicted std2s for each one of the test configurations
        C: ndarray(N,N)
            The covariance of the posterior distribution. It is used several times in the BO framework
        mu: ndarray(N,1)
            The mean of the posterior distribution. It is used several times in the BO framework
        """
        
         
        if xprime is not None:
            self.xprime = xprime

        kx_star = self.kernel_hyper(self.X, self.xprime, self.theta, show=show)

        if kx_star is None:
            if show: print('kx_star is None')
            return None

        kx = self.kernel_hyper(self.X, self.X, self.theta)
        if kx is None:
            if show: print('kx is None')
            return None

        if len(xprime.shape) > 1:
            m_xstar = self.xprime.mean(axis=1).reshape(-1, 1)
        else:
            m_xstar = self.xprime

        m_xstar = np.zeros(m_xstar.shape)

        m_const = self.m_const

        kx_inv = self.invers(kx)
        if kx_inv is None:
            if show: print('kx_inv is None')
            return None

        m_const = self.m_const
        
        if self.lg:
            Lambda, gamma = self.lambdaGamma(self.m_const, self.theta)
        else:
            Lambda, gamma = self.gammaLambda(self.m_const, self.theta)
        
        if Lambda is None or gamma is None:
            if show: print('Lambda is None or gamma is None')
            return None

        C_inv = kx_inv + Lambda

        C = self.invers(C_inv)
        if C is None:
            if show: print('C is None')
            return None

        self.C = C

        mu = self.m_const + np.dot(C, gamma)

        self.mu = mu

        mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, self.mu))
        
        #Now calculate the covariance
        kstar_star = self.kernel_hyper(self.xprime, self.xprime, self.theta)
        if kstar_star is None:
            if show: print('kstar_star is None')
            return None

        Lambda_inv = self.invers(Lambda)
        if Lambda_inv is None:
            if show: print('Lambda_inv is None')
            return None

        kx_lamdainv = kx + Lambda_inv

        kx_lamdainv_inv = self.invers(kx_lamdainv)

        if kx_lamdainv_inv is None:
            if show: print('kx_lamdainv_inv is None')
            return None

        cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
        
        std2 = np.diagonal(cov).reshape(-1, 1)

        self.activated = True
        self.asy_mean = mean

        if not full_cov:
            return mean, std2
        else:
            return mean, std2, cov


    def pred_old(self, conf_num, steps, fro=None):
        #Here conf_num is from 1 onwards. That's mu_n = mu[conf_num - 1, 0] in the for-loop
        if self.activated:
            yn = self.y_train[conf_num -1]
            if fro is None:
                t = np.arange(1, yn.shape[0] + 1)
                tprime = np.arange(yn.shape[0] + 1, yn.shape[0] + 1 + steps)
            else:
                t = np.arange(1, fro)
                tprime = np.arange(fro, fro + steps)

            mu = self.mu
            mu_n = mu[conf_num - 1, 0]
            C = self.C
            Cnn = C[conf_num - 1, conf_num - 1]

            yn = yn.reshape(-1, 1)

            ktn = self.kernel_curve(t, t, self.theta)
            if ktn is None:
                return None

            ktn_inv = self.invers(ktn)
            if ktn_inv is None:
                return None

            ktn_star = self.kernel_curve(t, tprime, self.theta)
            if ktn_star is None:
                return None

            Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T,
                                                           np.dot(ktn_inv, np.ones((t.shape[0], 1))))

            if yn.shape[0] > ktn_inv.shape[0]:
                yn = yn[:ktn_inv.shape[0]]

            mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + Omega*mu_n
            ktn_star_star = self.kernel_curve(tprime, tprime, self.theta)
            if ktn_star_star is None:
                return None

            cov = ktn_star_star - \
                np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + \
                np.dot(Omega, np.dot(Cnn, Omega.T))
            std2 = np.diagonal(cov).reshape(-1, 1)

            return mean, std2
        else:
            raise Exception


    def pred_new(self, steps=1, xprime=None, y=None, asy=False):
        """
        Params
        ------
        asy: Whether the asymptotic has already been calculated or not.
        """

        if xprime is not None:
            self.x_test = xprime

        if asy is False:
            asy_mean, std2star = self.pred_asympt(xprime[np.newaxis,:])
        else:
            asy_mean = self.asy_mean

        if type(asy_mean) is np.ndarray:
            asy_mean = asy_mean[0]

        fro = 1
        ts = np.arange(fro, (fro+1))
        k_ts_ts = self.kernel_curve(ts,ts, self.theta)

        return asy_mean, std2star + k_ts_ts


    def getYvector(self, y):
        """
        Transform the y_train from type ndarray(N, dtype=object) to ndarray(T, 1).
        That's necessary for doing matrices operations

        Returns
        -------
        y_vec: ndarray(T,1)
            An array containing all loss measurements of all training curves. They need
            to be stacked in the same order as in the configurations array x_train
        """
        y_vec = np.array([y[0]])
        for i in range(1, y.shape[0]):
            y_vec = np.append(y_vec, y[i])
        return y_vec.reshape(-1, 1)

    def getOmicron(self, y):
        """
        Caculates the matrix O = blockdiag(1_1, 1_2,...,1_N), a block-diagonal matrix, where each block is a vector of ones
        corresponding to the number of observations in its corresponding training curve

        Parameters
        ----------
        y: ndarray(N, dtype=object)
            All training curves stacked together

        Returns
        -------
        O: ndarray(T,N)
            Matrix O is used in several computations in the BO framework, specially in the marginal likelihood
        """
        O = block_diag(np.ones((y[0].shape[0], 1)))

        for i in range(1, y.shape[0]):
            O = block_diag(O, np.ones((y[i].shape[0], 1)))
        return O

    def kernel_hyper(self, x, xprime, theta, show=False):
        """
        Calculates the kernel for the GP over configuration hyperparameters

        Parameters
        ----------
        x: ndarray
            Configurations of hyperparameters, each one of shape D
        xprime: ndarray
            Configurations of hyperparameters. They could be the same or different than x, 
            depending on which covariace is being built

        Returns
        -------
        ndarray
            The covariance of x and xprime
        """

        flex = x.shape[-1]
        theta_d = np.exp(theta[:flex])
        if not self.samenoise:
            theta0, alpha, beta, noiseHyper, noiseCurve = np.exp(theta[flex:])
        else:
            theta0, alpha, beta, noise = np.exp(theta[flex:])
            noiseHyper = noiseCurve = noise

        if len(xprime.shape)==1:
            xprime = xprime.reshape(1,len(xprime))

        if len(x.shape)==1:
            x = x.reshape(1,len(x))

        try:
            r2 = np.sum(((x[:, np.newaxis] - xprime)**2) /
                theta_d**2, axis=-1)
            fiveR2 = 5 * r2
            result = theta0 *(1 + np.sqrt(fiveR2) + fiveR2/3.)*np.exp(-np.sqrt(fiveR2))
            if result.shape[1] > 1:
                toadd = np.eye(N=result.shape[0], M=result.shape[1])
                result = result +  toadd*noiseHyper
            return result
        except:
            return None

    def kernel_curve(self, t, tprime, theta):
        """
        Calculates the kernel for the GP over training curves

        Parameters
        ----------
        t: ndarray
            learning curve steps
        tprime: ndarray
            learning curve steps. They could be the same or different than t, depending on which covariace is being built

        Returns
        -------
        ndarray
            The covariance of t and tprime
        """

        flex = self.X.shape[-1]
        theta_d = np.exp(theta[:flex])
        if not self.samenoise:
            theta0, alpha, beta, noiseHyper, noiseCurve = np.exp(theta[flex:])
        else:
            theta0, alpha, beta, noise = np.exp(theta[flex:])
            noiseHyper = noiseCurve = noise

        try:
            result = np.power(beta, alpha) / \
                np.power(((t[:, np.newaxis] + tprime) + beta), alpha)

            result = result + \
                np.eye(N=result.shape[0], M=result.shape[1]) * noiseCurve

            return result
        except:
            return None

    def lambdaGamma(self, m_const, theta):
        """
        Difference here is that the cholesky decomposition is calculated just once for the whole Kt and thereafter
        we solve the linear system for each Ktn.
        """
        Kt = self.getKt(self.ys, theta)
        
        self.Kt_chol = self.calc_chol(Kt)
        if self.Kt_chol is None:
            return None, None
        dim = self.ys.shape[0]
        Lambda = np.zeros((dim, dim))
        gamma = np.zeros((dim, 1))
        index = 0
        for i, yn in enumerate(self.ys):
            lent = yn.shape[0]
            ktn_chol = self.Kt_chol[index:index + lent, index:index + lent]
            
            index += lent
            ktn_inv = self.inverse_chol(K=None, Chl=ktn_chol)
            if ktn_inv is None:
                return None, None
            one_n = np.ones((ktn_inv.shape[0], 1))
            
            Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
            gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))

        return Lambda, gamma

    def gammaLambda(self, m_const, theta):
        '''
        Calculates Lambda according to the following: Lamda = transpose(O)*inverse(Kt)*O
        = diag(l1, l2,..., ln) =, where ln = transpose(1n)*inverse(Ktn)*1n
        Calculates gamma according to the following: gamma = transpose(O)*inverse(Kt)*(y - Om),
        where each gamma element gamma_n = transpose(1n)*inverse(Ktn)*(y_n -m_n)
        
        Parameters
        ----------
        m_const: float
            the infered mean of f, used in the joint distribution of f and y.        
        
        Returns
        -------
        gamma: ndarray(N, 1)
            gamma is used in several calculations in the BO framework
        Lambda: ndarray(N, N)
                Lamda is used in several calculations in the BO framework
        '''
        dim = self.ys.shape[0]
        Lambda = np.zeros((dim, dim))
        gamma = np.zeros((dim, 1))
        index = 0

        for yn in self.ys:
            yn = yn.reshape(-1,1)
            t = np.arange(1, yn.shape[0]+1)
            
            ktn = self.kernel_curve(t, t, theta)

            ktn_inv = self.invers(ktn)
            one_n = np.ones((ktn.shape[0], 1))
            onenT_ktnInv = np.dot(one_n.T, ktn_inv)

            Lambda[index, index] = np.dot(onenT_ktnInv, one_n)
            gamma[index, 0] = np.dot(onenT_ktnInv, yn - m_const[index])            

            index+=1
        
        return Lambda, gamma

    def getKtn(self, yn, theta):
        t = np.arange(1, yn.shape[0] + 1)
        ktn = self.kernel_curve(t, t, theta)
        return ktn

    def getKt(self, y, theta):
        """
        Caculates the blockdiagonal covariance matrix Kt. Each element of the diagonal corresponds
        to a covariance matrix Ktn

        Parameters
        ----------
        y: ndarray(N, dtype=object)
            All training curves stacked together

        Returns
        -------
        """
        
        ktn = self.getKtn(y[0], theta)
        O = block_diag(ktn)

        for i in range(1, y.shape[0]):
            ktn = self.getKtn(y[i], theta)
            O = block_diag(O, ktn)

        return O

    def invers(self, K):
        if self.invChol:
            invers = self.inverse_chol(K)
        else:
            try:
                invers = np.linalg.inv(K)
            except:
                invers = None

        return invers


    def inverse_chol(self, K=None, Chl=None):
        """ 
        One can use this function for calculating the inverse of K through cholesky decomposition
        
        Parameters
        ----------
        K: ndarray
            covariance K
        Chl: ndarray
            cholesky decomposition of K

        Returns
        -------
        ndarray 
            the inverse of K
        """
        if Chl is not None:
            chol = Chl
        else:
            chol = self.calc_chol(K)

        if chol is None:
            return None

        inve = 0
        error_k = 1e-25
        while(True):
            try:
                choly = chol + error_k * np.eye(chol.shape[0])
                inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
                break
            except np.linalg.LinAlgError:
                error_k *= 10
        return inve

    def calc_chol(self, K):
        """
        Calculates the cholesky decomposition of the positive-definite matrix K

        Parameters
        ----------
        K: ndarray
            Its dimensions depend on the inputs x for the inputs. len(K.shape)==2

        Returns
        -------
        chol: ndarray(K.shape[0], K.shape[1])
            The cholesky decomposition of K
        """
        
        
        error_k = 1e-25
        chol = None
        once = False
        index = 0
        found = True
        while(index < 100):
            try:
                if once is True:
                    Ky = K + error_k * np.eye(K.shape[0])
                else:
                    Ky = K + error_k * np.eye(K.shape[0])
                    once = True
                chol = np.linalg.cholesky(Ky)
                found = True
                break
            except np.linalg.LinAlgError:
                error_k *= 10
                found = False
            
            index += 1
        if found:
            return chol
        else:
            return None

    def nplog(self, val, minval=0.0000000001):
        return np.log(val.clip(min=minval)).reshape(-1, 1)
    
