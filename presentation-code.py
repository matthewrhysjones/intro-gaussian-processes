# %%

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np 
from numpy.random import multivariate_normal
import random

import kernels
import models

# %% some sample data

np.random.seed(20)
x = np.linspace(0,1,100)[:,None]

hyps = np.array([1,.1,1e-6])
kernel = kernels.SquaredExponential(x,x,hyps)
k, _ , _ = kernel.compute_kernel()

mu = np.zeros((100,))
nsamples = 1
y = multivariate_normal(mu, k,nsamples).T

npts = 10
random.seed(1)
randidx = random.sample(range(0,100),npts)

xsample = x[randidx]
ysample = y[randidx]

plt.figure()
plt.plot(xsample,ysample,'kx')

# %% fit a GP to sample data (known hyps in this case)

D = (xsample,ysample)
mu_pred, cov_pred = models.predict(hyps,D,x,kernels.SquaredExponential)
var_pred = np.diag(cov_pred)

plt.figure()
plt.plot(xsample,ysample,'kx',label ='train pts')
plt.plot(x,mu_pred, label ='GP predictions')
plt.fill_between(x[:,0], mu_pred[:,0] - 3*np.sqrt(var_pred), mu_pred[:,0] + 3*np.sqrt(var_pred),color = 'darkgreen', alpha = 0.3, label ='confidence bounds')
plt.legend();

# %% look at some GP samples (function samples) from GP posterior

nsamples = 10
gp_samples = multivariate_normal(mu, cov_pred, nsamples).T
gp_samples += mu_pred

plt.figure()
plt.plot(x,gp_samples)
plt.plot(xsample,ysample,'kx',label ='train pts')
plt.title('Posterior GP samples')
plt.legend();

# %% Gaussian visualisation

#initialize a normal distribution with frozen in mean=-1, std. dev.= 1
gaussian1 = norm(loc = 0., scale = 1.0)

x = np.arange(-10, 10, .1)

#plot the pdfs of these normal distributions 
plt.figure()
plt.plot(x, gaussian1.pdf(x));
plt.ylabel('p(x)');
plt.xlabel('x');

gaussian2 = norm(loc = 0., scale = 1)
gaussian3 = norm(loc = 2, scale = 2.5)

plt.figure();
plt.plot(x, gaussian2.pdf(x), x, gaussian3.pdf(x));
plt.ylabel('p(x)');
plt.xlabel('x');

# %% 2D Gaussian

mu = np.array([0,0])
cov_ind = np.array([[1,0], [0,1]])
cov_cor = np.array([[1,0.85],[0.85,1]])

samples_ind = multivariate_normal(mu,cov_ind, size = 1000)
samples_cor = multivariate_normal(mu,cov_cor, size = 1000)

plt.figure();
fig, axs = plt.subplots(1,2, figsize=(10,6));
axs[0].plot(samples_ind[:,0], samples_ind[:,1],'x', color = 'indigo');
axs[1].plot(samples_cor[:,0], samples_cor[:,1],'x', color = 'indigo');
plt.setp(axs, xlim=(-4,4), ylim=(-4,4));

fig.add_subplot(111, frameon=False);
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False);
plt.xlabel("x1");
plt.ylabel("x2");

# %% Beyond 2D

np.random.seed(20)
x = np.linspace(0,1,100)[:,None]

hyps = np.array([1,.25])
kernel = kernels.SquaredExponential(x,x,hyps)
k, _ , _ = kernel.compute_kernel()

mu = np.zeros((100,))
nsamples = 5
y = multivariate_normal(mu, k,nsamples).T

d = 2
ysample = y[0::100//d,:]
index = np.linspace(1,d,d)

plt.figure()
plt.plot(index,ysample)

d = 4
ysample = y[0::100//d,:]
index = np.linspace(1,d,d)

plt.figure()
plt.plot(index,ysample)

d = 5
ysample = y[0::100//d,:]
index = np.linspace(1,d,d)

plt.figure()
plt.plot(index,ysample)

d = 10
ysample = y[0::100//d,:]
index = np.linspace(1,d,d)

plt.figure()
plt.plot(index,ysample)

d = 20
ysample = y[0::100//d,:]
index = np.linspace(1,d,d)

plt.figure()
plt.plot(index,ysample)

d = 50
ysample = y[0::100//d,:]
index = np.linspace(1,d,d)

plt.figure()
plt.plot(index,ysample)

# %% kernels

hyps = [1,0.5]

kernelSE = kernels.SquaredExponential(x,x,hyps)
kernelMA12 = kernels.Matern12(x,x,hyps)

kSE, _ , _ = kernelSE.compute_kernel()
kMA12, _ , _ = kernelMA12.compute_kernel()

mu = np.zeros((100,))
nsamples = 10

ySE = multivariate_normal(mu, kSE,nsamples).T
yMA12 = multivariate_normal(mu, kMA12,nsamples).T

plt.figure();
fig, axs = plt.subplots(1,2, figsize=(10,6));
axs[0].plot(x,ySE)
axs[0].set_title('Samples from SE kernel');
axs[1].plot(x,yMA12)
axs[1].set_title('Samples from Matern 1/2 kernel');

hyps = [1,0.5]

kernelMA32 = kernels.Matern32(x,x,hyps)
kernelMA12 = kernels.Matern12(x,x,hyps)

kMA32, _ , _ = kernelMA32.compute_kernel()
kMA12, _ , _ = kernelMA12.compute_kernel()

mu = np.zeros((100,))
nsamples = 10

yMA12 = multivariate_normal(mu, kMA12,nsamples).T
yMA32 = multivariate_normal(mu, kMA32,nsamples).T

plt.figure();
fig, axs = plt.subplots(1,2, figsize=(10,6));
axs[0].plot(x,yMA12)
axs[0].set_title('Samples from Matern 1/2 kernel');
axs[1].plot(x,yMA32)
axs[1].set_title('Samples from Matern 3/2 kernel');

hyps = [1,0.5,0.5]

kernelPeriodic = kernels.Periodic(x,x,hyps)

kPeriodic, _ , _ = kernelPeriodic.compute_kernel()

mu = np.zeros((100,))
nsamples = 10

yPeriodic = multivariate_normal(mu, kPeriodic,nsamples).T

plt.figure()
plt.plot(x,yPeriodic)
plt.title('Samples from periodic kernel');

# %%

x1, y1 = 2, 3
x2, y2 = 6, 7
eqx, eqy = 4,3.5

middle_x = (x1 + x2) / 2
middle_y = (y1 + y2) / 2
distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Plot the two points and the line connecting them
plt.figure()
label_offset = 0.25
plt.plot([x1, x2], [y1, y2], 'go-',markersize = 8)
plt.text(x1 + 0.5*label_offset, y1 + 1.5*label_offset, f'Point 1 \n({x1},{y1})', ha='center')
plt.text(x2 - 1.8*label_offset, y2 - 1*label_offset, f'Point 2 \n({x2},{y2})', ha='center')

plt.text(middle_x, middle_y + 1.5*label_offset, f'd', ha='center', va='top')
plt.text(eqx, eqy, 'd = $\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$', ha = 'left')

# Add labels and title
plt.xlabel('x');
plt.ylabel('y');

# %% hyperparameters 

hyps_long = [1,1] # long lengthscale
hyps_short = [1,0.1] # short lenthscale

kernelSE_long = kernels.SquaredExponential(x,x,hyps_long)
kernelSE_short = kernels.SquaredExponential(x,x,hyps_short)

kSElong, _ , _ = kernelSE_long.compute_kernel()
kSEshort, _ , _ = kernelSE_short.compute_kernel()

mu = np.zeros((100,))
nsamples = 10

ySElong = multivariate_normal(mu, kSElong,nsamples).T
ySEshort = multivariate_normal(mu, kSEshort,nsamples).T

plt.figure()
fig, ax = plt.subplots(1,2)
ax[0].plot(x,ySElong)
ax[0].set_title('lengthscale = 1');
ax[1].plot(x,ySEshort)
ax[1].set_title('lengthscale = 0.1');

# %% overview of fitting GP: 1) trainining data (& standardisation)
# 2) select kernel 3) fit hyps 4) make predictions on test data that the model hasnt seen

np.random.seed(20)

x = np.linspace(0,2,100)[:,None]
y = np.sin(2*x)**2 * 0.5 * np.cos(2*x)
y += 1e-3 * np.random.normal(0,1, size = (len(y),1))

random.seed(20)
npts = 15
randidx = random.sample(range(0,100),npts)

xtrain = x[randidx]
ytrain = (y[randidx] - np.mean(y[randidx]))/np.std(y[randidx])

D = (xtrain,ytrain)

N = len(xtrain) # num training points 

xtest = x
ytest = (y - np.mean(y[randidx]))/np.std(y[randidx])

plt.figure()
plt.plot(xtest,ytest)
plt.plot(xtrain,ytrain,'x', label = 'training pts')
plt.legend();

kernel = kernels.SquaredExponential

hyps0 = np.array([0.1,0.1,0])
hyps_opt, nlml = models.train(hyps0,D,kernel)
print(hyps_opt)
print(nlml);

mu_pred, cov_pred = models.predict(hyps_opt,D,xtest,kernel)
var_pred = np.diag(cov_pred)

plt.figure()
plt.plot(xtest,mu_pred,label = 'pred')
plt.fill_between(xtest[:,0], (mu_pred + 3*np.sqrt(var_pred))[:,0], (mu_pred - 3*np.sqrt(var_pred))[:,0] ,color ='darkgreen', alpha = 0.3,label = 'confidence')
plt.plot(xtrain,ytrain,'kx',label = 'train')
plt.plot(xtest,ytest,':',label = 'true')
plt.legend();

# %%