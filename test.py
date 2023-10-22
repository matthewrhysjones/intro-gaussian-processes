# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

toy_func = lambda x: np.sin(10*x)**2 + np.cos(10*x) + 5*x

#Xall = np.random.uniform(0,10,100)[:,None]
Xall = np.linspace(0,1,100)[:,None]
Yall = toy_func(Xall) + np.random.normal(0,10e-2, size = (len(Xall),1))

plt.figure()
plt.plot(Xall,Yall,'x')

# %%

xtrain = Xall[:,:]
ytrain = Yall[:,:]
ytrain = (ytrain - np.mean(ytrain))/np.std(ytrain)

xtest = Xall
ytest = Yall
# %%

from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from numpy.linalg import solve, cholesky

def squared_exponential(ins1,ins2,hyps):

    sf2 = hyps[0]
    ll = hyps[1]
    sn2 = hyps[2]

    dist = cdist(ins1,ins2,'euclidean')
    
    return sf2*(np.exp(-dist**2/(2*ll**2)))

def matern32(ins1,ins2,hyps):

    sf2 = hyps[0]
    ll = hyps[1]
    sn2 = hyps[2]

    dist = cdist(ins1,ins2,'euclidean')
    
    return sf2*(1+(np.sqrt(3)*dist)/ll)*np.exp((-np.sqrt(3)*dist)/ll)

def loss_function(hyps,xtrain,ytrain):

    hyps = np.exp(hyps)

    Kxx = squared_exponential(xtrain,xtrain,hyps) +  hyps[2]*np.eye((len(xtrain)))
    #Kxxinv = np.linalg.inv(Kxx + 1e-8*np.eye((Kxx.shape[0])))
    L = cholesky(Kxx + 1e-6*np.eye((Kxx.shape[0])))
    alpha = solve(L.T,solve(L,ytrain))

    lml = -0.5*ytrain.T.dot(alpha) - 0.5*np.sum(np.log(np.diag(L))) - (len(xtrain)/2*np.log(2*np.pi))
    #lml = -0.5*ytrain.T.dot(Kxxinv).dot(ytrain) - 0.5*np.log(np.linalg.det(Kxx)) - (len(xtrain)/2 * np.log(2*np.pi))

    return -lml

def train(xtrain,ytrain):

    # Kxxstar = squared_exponential(xtrain,xtest)
    # Kxstarxstar = squared_exponential(xtest,xtest)

    hyps0 = np.random.uniform(0,2,3)
    hyps0[2] = hyps0[2]*1e-6

    opt_res = minimize(loss_function,np.exp(hyps0),
    args = (xtrain,ytrain), method = 'L-BFGS-B')

    return opt_res


# %%

opt_post = train(xtrain,ytrain)
print(opt_post)
print(np.exp(opt_post['x']))
# %%
loss_function(hyps0,xtrain,ytrain)
# %%
