""" Oja's algorithm as described in ... ADD REFERENCE """
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
        step_size = self.hyperparameters['learning_rate']**0.5
        # update running average
        self.parameters['mean'] = \
                (self.parameters['t'] - 1.0)*self.parameters['mean'] \
                + point
        self.parameters['mean'] /= self.parameters['t'] 
        point -= self.parameters['mean']
        gradient = np.matmul(np.outer(point,point),self.parameters['U'])
        tmp = self.parameters['U'] + step_size * gradient
        self.parameters['U'], _ = np.linalg.qr(V,mode='reduced')

    def transform(self,points):
        return np.matmul(points,self.parameters['U'])

    def loss(self,points):
        uut = np.matmul(self.parameters['U'],self.parameters['U'].T)
        residuals = points - np.matmul(points,uut)
        loss = np.sum(np.power(residuals,2))
        return loss
