""" generate 2 views according to the probabilistic model of
Bach and Jordan, A Probabilistic Interpretation of Canonical Correlation Analysis, 2005. """
import numpy as np

# dimensionality of X, dimensionality of Y, dimensionality of latent variable
#dx,dy,k=110,214,56
dx,dy,k=11,21,5
num_samples=int(1e4)

# means
mu_x=np.random.uniform(low=0.0, high=1.0, size=(dx,1))
mu_y=np.random.uniform(low=-0.5, high=0.5, size=(dy,1))

# linear transformation matrices
W_x=np.random.uniform(low=-10.0, high=10.0, size=(dx,k))
W_y=np.random.uniform(low=-5.0, high=8.0, size=(dy,k))

# covariances
Psi_x_sqrt=np.random.uniform(low=-10.0, high=10.0, size=(dx,dx))
Psi_y_sqrt=np.random.uniform(low=-25.0, high=5.0, size=(dy,dy))

# sample latent variable
Z=np.random.normal(size=(k,num_samples))
# sample X
H_x=np.random.normal(size=(dx,num_samples))
X=mu_x + np.matmul(W_x,Z) + np.matmul(Psi_x_sqrt,H_x)
# sample Y
H_y=np.random.normal(size=(dy,num_samples))
Y=mu_y + np.matmul(W_y,Z) + np.matmul(Psi_y_sqrt,H_y)

# print out
np.savetxt('synth_data_bach_jordan_view1.csv',X.T,delimiter=',')
np.savetxt('synth_data_bach_jordan_view2.csv',Y.T,delimiter=',')
