import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ErrorMeanVarianceEstimator():
    def __init__(self,ddof=1,v_0=0.1):
        self.ddof = ddof
        self.v_0 = round(v_0,4)
        self.v_1 = None

    def estimate_mean(self,curvems):
        for errms in curvems:
            errms.mean = round(np.mean(errms.test_errors),4)

    def estimate_variance(self,curvems):
        for errms in curvems:
            if errms.num_ms < 2:
                continue

            errms.variance = round(np.var(errms.test_errors,ddof=self.ddof),4)

    def estimate_smooth_variance(self,curvems):
        filtered_errms = [
            errms for errms in curvems if errms.variance is not None]
        x = np.array([1/errms.num_train_samples for errms in filtered_errms])
        v = np.array([errms.variance - self.v_0 for errms in filtered_errms])
        w = np.array([errms.num_ms-1 for errms in filtered_errms])
        
        self.v_1 = round(np.sum(w*v*x) / np.sum(w*x*x),4)
        
        for errms in curvems:
            errms.smoothed_variance = self.get_smoothed_variance(
                errms.num_train_samples)

    def get_smoothed_variance(self,num_train_samples):
        return round(self.v_0 + (self.v_1 / num_train_samples),4)

    def estimate(self,curvems):
        self.estimate_mean(curvems)
        self.estimate_variance(curvems)
        self.estimate_smooth_variance(curvems)
    
    def visualize(self,curvems):
        filtered_errms = [
            errms for errms in curvems if errms.variance is not None]
        ns = [errms.num_train_samples for errms in filtered_errms]
        variances = [errms.variance for errms in filtered_errms]
        smoothed_variances = [errms.smoothed_variance \
            for errms in filtered_errms]

        plt.style.use('seaborn-whitegrid')
        plt.plot(ns,variances,marker='o',label='Sample Variance')
        plt.plot(ns,smoothed_variances,marker='o',label='Smoothed Variance')
        plt.legend()
        plt.xlabel('Number of training samples')
        plt.ylabel('Variance')