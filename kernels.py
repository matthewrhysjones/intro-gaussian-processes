import numpy as np 
from scipy.spatial.distance import cdist

class BaseKernel:

    def __init__(self,xtrain,xtest,hyps):

        self.hyps = hyps
        self.xtrain = xtrain
        self.xtest = xtest

        self.pairwise_xx = cdist(xtrain,xtrain)
        self.pairwise_xt = cdist(xtrain,xtest)
        self.pairwise_tt = cdist(xtest,xtest)
    
class SquaredExponential(BaseKernel):

    def k(self, d):

        sf2 = self.hyps[0]
        l = self.hyps[1]

        return sf2 * np.exp(-d**2 / (2 * l**2))

    def compute_kernel(self):

        kxx = self.k(self.pairwise_xx)
        kxt = self.k(self.pairwise_xt)
        ktt = self.k(self.pairwise_tt)

        return kxx, kxt, ktt
        
class Matern12(BaseKernel):
        
    def k(self,d):

        sf2 = self.hyps[0]
        l = self.hyps[1]

        return sf2 * np.exp(-d/l)
    
    def compute_kernel(self):

        kxx = self.k(self.pairwise_xx)
        kxt = self.k(self.pairwise_xt)
        ktt = self.k(self.pairwise_tt)

        return kxx, kxt, ktt

class Matern32(BaseKernel):
        
    def k(self,d):

        sf2 = self.hyps[0]
        l = self.hyps[1]

        return sf2 * (1+(np.sqrt(3)*d/l)) * np.exp(-np.sqrt(3)*d/l)
    
    def compute_kernel(self):

        kxx = self.k(self.pairwise_xx)
        kxt = self.k(self.pairwise_xt)
        ktt = self.k(self.pairwise_tt)      

        return kxx, kxt, ktt

class Periodic(BaseKernel):

    def k(self,d):

        sf2 = self.hyps[0]
        l = self.hyps[1]
        p = self.hyps[2]

        return  sf2 * np.exp(-2/l**2 * np.sin(np.pi*np.abs(d)/p)**2)
    
    def compute_kernel(self):

        kxx = self.k(self.pairwise_xx)
        kxt = self.k(self.pairwise_xt)
        ktt = self.k(self.pairwise_tt)

        return kxx, kxt, ktt


