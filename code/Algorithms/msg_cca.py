""" Matrix Stochastic Gradient method for Canonical Correlation Analysis as described in 
Raman Arora, Poorya Mianjy, and Teodor Marinov. Stochastic optimization for multiview representation learning using partial least squares. In International Conference on Machine Learning, pages 1786–1794, 2016."""

import numpy as np

class MSG_CCA:
    def __init__(self,hyperparameters):
        self.hyperparameters=hyperparameters
        assert 'dx' in hyperparameters, 'dimensionality of first view not specified'
        assert 'dy' in hyperparameters, 'dimensionality of second view not specified'
        assert 'k' in hyperparameters, 'dimensionality of projection not specified'
        assert 'learning_rate' in hyperparameters, \
                'initial learning rate not specified'
        assert self.hyperparameters['dx'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of original space'
        assert self.hyperparameters['dy'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of original space'
        self.parameters={
                'M': np.zeros(self.hyperparameters['dx'], self.hyperparameters['dy']),
                'M_bar': np.zeros(self.hyperparameters['dx'], self.hyperparameters['dy']),
                'mean_x': np.zeros(self.hyperparameters['dx']),
                'mean_y': np.zeros(self.hyperparameters['dy']),
                't': 0
                }

    def step(self,point):
        x,y=point
        self.parameters['t'] += 1
        # update running averages
        alpha = 1.0/self.parameters['t']
        self.parameters['mean_x'] = (1-alpha)*self.parameters['mean_x'] + alpha*x
        self.parameters['mean_y'] = (1-alpha)*self.parameters['mean_y'] + alpha*y
        x -= self.parameters['mean_x']
        y -= self.parameters['mean_y']
        gradient = np.outer(x,y)
        tmp = self.parameters['M'] + self.hyperparameters['learning_rate'] * gradient
        self.parameters['M'] = self.projection(tmp)
        self.parameters['M_bar'] = (1-alpha)*self.parameters['M_bar'] \
                + alpha*self.parameters['M']

    def projection(self,mat):
        """ projection of a matrix onto the feasible set
        the projection proj has the same singular vectors of mat, and its singular values are
        sigma(proj)_i = max(0, min(1, sigma(mat)_i + S)) where S is chosen such that
        \sum_i sigma(proj)_i = k
        Algorithm 2 in Raman Arora, Andy Cotter, and Nati Srebro. Stochastic optimization 
        of pca with capped msg. In Advances in Neural Information Processing Systems,
        pages 1815–1823, 2013."""


    def transform(self,points):
        return np.matmul(points,self.parameters['U'])

    def loss(self,points):
        points = points - np.expand_dims(self.parameters['mean'],axis=0)
        utx = np.matmul(points,self.parameters['U'])
        residuals = points - np.matmul(self.parameters['U'],utx.T).T
        loss = np.sum(np.power(residuals,2))
        return loss
