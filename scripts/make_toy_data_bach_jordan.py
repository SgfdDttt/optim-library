""" generate 2 views according to the probabilistic model of
Bach and Jordan, A Probabilistic Interpretation of Canonical Correlation Analysis, 2005. """
import numpy as np

# dimensionality of X, dimensionality of Y, dimensionality of latent variable
dx,dy,k=11,14,5
num_samples=int(1e6)

# means
mu_x=np.random.uniform(low=0.0, high=1.0, size=(dx,1))
mu_y=np.random.uniform(low=-0.5, high=0.5, size=(dy,1))

# linear transformation matrices
W_x=np.random.uniform(low=-10.0, high=10.0, size=(dx,k))
W_y=np.random.uniform(low=-5.0, high=8.0, size=(dy,k))

# covariances
Psi_x=np.random.uniform(low=-10.0, high=10.0, size=(dx,dx))
Psi_x=np.matmul(Psi_x,Psi_x.T) # so it's semi-def pos
Psi_y=np.random.uniform(low=-10.0, high=10.0, size=(dy,dy))
Psi_y=np.matmul(Psi_y,Psi_y.T) # so it's semi-def pos

# sample latent variable
Z=np.random.normal(size=(num_samples,k))

# sample both views
X,Y=[],[]
for ii in range(num_samples):
    # sample X
    mean_x=np.matmul(W_x,Z[ii:ii+1,:].T)+mu_x
    X.append(np.random.multivariate_normal(np.squeeze(mean_x),Psi_x))
    # sample Y
    mean_y=np.matmul(W_y,Z[ii:ii+1,:].T)+mu_y
    Y.append(np.random.multivariate_normal(np.squeeze(mean_y),Psi_y))
    assert False
# print out
