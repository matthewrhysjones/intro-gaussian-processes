# %%

import numpy as np
import scipy.stats as stats 
import matplotlib.pyplot as plt 
from numpy.random import multivariate_normal
import random

from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve

import kernels

# %%
np.random.seed(20)

x = np.linspace(0,1,100)[:,None]

hyps = np.array([1,.25])

kernel = kernels.SquaredExponential(x,x,hyps)
k, _ , _ = kernel.compute_kernel()

d = 100
N = d
ksample = k[0::100//d,0::100//d]
musample = np.zeros((d,))

ysample = multivariate_normal(musample, ksample,1).T[:,0] + 1e-3 * np.random.normal(0,1, size = (N,))

plt.plot(ysample)
# %%
random.seed(10)

npts = 30
randidx = random.sample(range(0,100),npts)

plt.figure()
plt.plot(x,ysample)
plt.plot(x[randidx],ysample[randidx],'x')

xtrain = x[randidx]
ytrain = (ysample[randidx] - np.mean(ysample[randidx]))/np.std(ysample[randidx])

N = len(xtrain)

xtest = x
ytest = (ysample - np.mean(ysample[randidx]))/np.std(ysample[randidx])

hyps0 = np.array([1,1])

kernel = kernels.SquaredExponential(xtrain,xtest,hyps0)
kxx,kxt,ktt = kernel.compute_kernel()

L = np.linalg.cholesky(kxx + 1e-6*np.eye(N))
alpha = np.linalg.solve(L.T,np.linalg.solve(L,ytrain))
mu_pred = kxt.T @ alpha

v = np.linalg.solve(L,kxt)
var_pred = np.diag(ktt - v.T @ v)

plt.figure()
plt.plot(xtest,mu_pred)
plt.plot(xtest,mu_pred + 3*np.sqrt(var_pred), color = 'darkgreen', linestyle = '--', alpha = 0.6)
plt.plot(xtest,mu_pred - 3*np.sqrt(var_pred), color = 'darkgreen', linestyle = '--', alpha = 0.6)
plt.plot(xtrain,ytrain,'x')

# %%

def loss(hyps,D,kernel):

    jitter = 1e-6

    xtrain = D[0]
    ytrain = D[1]

    N = len(ytrain)

    hyps = np.exp(hyps)

    sn2 = hyps[-1]

    kxx, _ , _ = kernel(xtrain,xtrain,hyps).compute_kernel()
    kxx += (sn2 * np.eye(N))

    L = np.linalg.cholesky(kxx + jitter*np.eye(N))
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,ytrain)) #cho_solve((L, True), ytrain) 

    nlml = (0.5 * ytrain.T @ alpha) + np.sum(np.log(np.diag(L))) + (0.5 * N * np.log(2*np.pi))

    return nlml
    
def train(hyps,D,kernel):

    opt = minimize(loss, hyps, args =(D, kernel), method = 'L-BFGS-B')
    hyps_opt = np.exp(opt.x)
    nlml = opt.fun

    return hyps_opt, nlml
    
def predict(hyps,D,xtest,kernel):

    xtrain = D[0]
    ytrain = D[1]
    N = len(ytrain)

    jitter = 1e-6
    sn2 = hyps[-1]

    kxx, kxt, ktt = kernel(xtrain,xtest,hyps).compute_kernel()
    kxx += sn2*np.eye(N)

    L = np.linalg.cholesky(kxx + jitter*np.eye(N))
    alpha = np.linalg.solve(L.T,np.linalg.solve(L,ytrain))

    v = np.linalg.solve(L,kxt)

    mu_pred = kxt.T @ alpha
    var_pred = np.diag(ktt - v.T @ v)

    return mu_pred, var_pred

# %%

hyps0 = np.array([1,0.1,1e-4])
D =(xtrain,ytrain)
xtest = xtest 
kernel = kernels.SquaredExponential

hyps_opt, nlml = train(hyps0,D,kernel)
print(hyps_opt)
print(nlml)

mu_pred, var_pred = predict(hyps_opt,D,xtest,kernel)

# %%

plt.figure()
plt.plot(xtrain,ytrain,'x')
plt.plot(xtest,ytest,'--')
plt.plot(xtest,mu_pred)
plt.fill_between(xtest[:,0], (mu_pred + 3*np.sqrt(var_pred)), (mu_pred - 3*np.sqrt(var_pred)) ,color ='darkgreen', alpha = 0.3)

# %%
