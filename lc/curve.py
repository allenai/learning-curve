import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .variance import ErrorMeanVarianceEstimator


class LearningCurve():
    def __init__(self,param,covariance,gamma):
        self.param = param
        self.covariance = covariance
        self.gamma = gamma

    @property
    def alpha(self):
        return round(self.param[0],4)

    @property
    def eta(self):
        return round(self.param[1],4)

    def beta(self,N):
        return round(-2*self.eta*self.gamma*N**self.gamma,4)

    def __call__(self,num_train_samples):
        return round(self.alpha + self.eta*num_train_samples**self.gamma,4)
    
    def summary(self,N):
        return {
            f'error_{N}': self(N),
            f'beta_{N}': self.beta(N),
            'gamma': self.gamma,
            'alpha': self.alpha,
            'eta': self.eta,
        }

    def print_summary(self,N):
        print('-'*30)
        print('Learning curve summary')
        print('-'*30)
        for k,v in self.summary(N).items():
            print(f'{k}: {v}')


class LearningCurveEstimator():
    def __init__(self,cfg):
        self.cfg = cfg
        self.err_mean_var_estimator = ErrorMeanVarianceEstimator(
            self.cfg.ddof, 
            self.cfg.v_0)

    def compute_A(self,curvems,gamma):
        A = []
        for errms in curvems:
            n = errms.num_train_samples
            for i in range(errms.num_ms):
                A.append([1, n**gamma])
        
        return np.array(A)
    
    def compute_weights(self,curvems):
        wts = []
        for errms in curvems:
            err_var = errms.fetch_variance(self.cfg.variance_type)
            w = 1/(err_var*errms.num_ms)
            wts.extend([w]*errms.num_ms)
        
        return np.array(wts)

    def gather_all_errors(self,curvems):
        e = []
        for errms in curvems:
            e.extend(errms.test_errors)

        return np.array(e)

    def gather_all_error_variances(self,curvems):
        v = []
        for errms in curvems:
            err_var = errms.fetch_variance(self.cfg.variance_type)
            v.extend([err_var]*errms.num_ms)

        return np.array(v)

    def estimate_given_gamma(self,curvems,gamma=None):
        if gamma is None:
            gamma = self.cfg.gamma

        self.err_mean_var_estimator.estimate(curvems)
        A = self.compute_A(curvems,gamma)
        wts = self.compute_weights(curvems)
        W_half = np.diag(wts**0.5)
        M = np.linalg.pinv(W_half @ A) @ W_half
        e = self.gather_all_errors(curvems)
        v = self.gather_all_error_variances(curvems)
        S = np.diag(v)

        lc_params = M @ e
        lc_covariance = M @ S @ M.T
        learning_curve = LearningCurve(lc_params,lc_covariance,gamma)

        e_pred = A @ lc_params
        agg_method = np.sum
        if self.cfg.normalize_objective is True:
            agg_method = np.mean

        objective = round(agg_method(wts * (e - e_pred)**2),4)
        
        return learning_curve, objective

    def estimate(self,curvems):
        if self.cfg.gamma_search is False:
            return self.estimate_given_gamma(curvems)

        lc_objs = []
        for gamma in np.arange(*self.cfg.gamma_range,0.01):
            gamma = round(gamma,2)
            lc,obj = self.estimate_given_gamma(curvems,gamma)
            search_obj = obj + self.cfg.search_reg_coeff * np.abs(gamma+0.5)
            lc_objs.append((lc,obj,search_obj))

        lc,obj,_ = min(lc_objs,key=lambda x: x[2])
        return lc, obj

    def plot(self,learning_curve,curvems,label,color='r',linestyle='-'):
        plt.rcParams.update({
            "text.usetex": True})
        plt.style.use('seaborn-whitegrid')

        # Plot measurements
        if curvems is not None:
            errs = []
            xs = []
            for errms in curvems:
                errs.extend(errms.test_errors)
                xs.extend([errms.num_train_samples**-0.5]*errms.num_ms)

            plt.scatter(xs,errs,marker='o',c=color,s=self.cfg.marker_size,
                linewidths=0.5,edgecolors='k',zorder=3)

        # plot curve
        max_x = self.cfg.min_n**-0.5
        xs = np.arange(1e-4, max_x, max_x / self.cfg.num_interp_pts)
        ns = xs**-2
        errs = [learning_curve(n) for n in ns]
        N = self.cfg.N
        eN = round(learning_curve(N),2)
        betaN = round(learning_curve.beta(N),2)
        gamma = learning_curve.gamma 
        plt_label = rf'{label} $(e_{{{N}}}={eN}; \; ' + \
            rf'\beta_{{{N}}}={betaN}; \; ' + \
            rf'\gamma={gamma})$'
        plt.plot(xs,errs,linestyle,color=color,label=plt_label,zorder=2)

        plt.legend()

        # add axis labels and ticks
        plt.xlim(0,max_x)
        plt.ylim(0,100)
        xlocs, xlabels = plt.xticks() 
        xlocs = [round(x,3) for x in xlocs if x!=0]
        ns = [math.ceil(1/float(x)**2) for x in xlocs]
        xlabels = [f'{x}\n({n})' for x,n in zip(xlocs,ns)]
        xlocs = [0] + xlocs
        xlabels = ['0\n('+r'$\infty$'+')'] + xlabels
        plt.xticks(xlocs,xlabels)
        plt.xlabel(r'$n^{-0.5}$')
        plt.ylabel('Error')

        # mark the 4x full data checkpoint
        x = (4*self.cfg.N)**-0.5
        plt.axvline(x=x,zorder=1,linewidth=0.5,color='k',linestyle='--')