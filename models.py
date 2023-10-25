import numpy as np
from scipy.optimize import minimize

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
    cov_pred = ktt - v.T @ v


    return mu_pred, cov_pred
