""" Oja's algorithm as described in
Zeyuan Allen-Zhu and Yuanzhi Li. First efficient convergence for streaming k-pca: a global, gap-free, and near-optimal rate. In Foundations of Computer Science (FOCS), 2017 IEEE 58th Annual Symposium on, pages 487–492. IEEE, 2017. """
import numpy as np

class Oja:
    def __init__(self,hyperparameters):
        self.hyperparameters=hyperparameters
        assert 'd' in hyperparameters, 'dimensionality of input vectors not specified'
        assert 'k' in hyperparameters, 'dimensionality of projection not specified'
        assert 'learning_rate' in hyperparameters, \
                'initial learning rate not specified'
        assert self.hyperparameters['d'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of original space'
        self.parameters={
                'U': np.eye(self.hyperparameters['d'], self.hyperparameters['k']),
                'mean': np.zeros(self.hyperparameters['d']),
                't': 0
                }

    def step(self,point):
        self.parameters['t'] += 1
        step_size = self.hyperparameters['learning_rate']\
                *(self.parameters['t']**(-0.5))
        # update running average
        alpha = 1.0/self.parameters['t']
        self.parameters['mean'] = \
                (1-alpha)*self.parameters['mean'] \
                + alpha*point
        point -= self.parameters['mean']
        gradient = np.outer(point,np.matmul(point.T,self.parameters['U']))
        tmp = self.parameters['U'] + step_size * gradient
        self.parameters['U'], _ = np.linalg.qr(tmp,mode='reduced')

    def transform(self,points):
        return np.matmul(points,self.parameters['U'])

    def loss(self,points):
        points = points - np.expand_dims(self.parameters['mean'],axis=0)
        utx = np.matmul(points,self.parameters['U'])
        residuals = points - np.matmul(self.parameters['U'],utx.T).T
        loss = np.sum(np.power(residuals,2))
        return loss
