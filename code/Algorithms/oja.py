""" Oja's algorithm as described in
Zeyuan Allen-Zhu and Yuanzhi Li. First efficient convergence for streaming k-pca: a global, gap-free, and near-optimal rate, 2017."""
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
        self.hyperparameters.setdefault('mean_center',1.0)
        assert self.hyperparameters['mean_center'] in [0.0, 1.0], \
                "either mean center or don't"
        self.parameters={
                #'U': np.eye(self.hyperparameters['d'], self.hyperparameters['k']),
                'U': np.random.uniform(low=-1., high=1., \
                        size=(self.hyperparameters['d'], self.hyperparameters['k']) ),
                'mean': np.zeros(self.hyperparameters['d']),
                't': 0
                }
        self.parameters['U'], _ = np.linalg.qr(self.parameters['U'],mode='reduced')

    def step(self,point):
        self.parameters['t'] += 1
        step_size = self.hyperparameters['learning_rate']\
                /(self.parameters['t']**0.5)
        # update running average
        alpha = 1.0/self.parameters['t']
        self.parameters['mean'] = \
                (1-alpha)*self.parameters['mean'] \
                + alpha*point
        point -= self.hyperparameters['mean_center']*self.parameters['mean']
        utx=np.matmul(point.T,self.parameters['U'])
        gradient = np.outer(point,np.matmul(point.T,self.parameters['U']))
        tmp = self.parameters['U'] + step_size * gradient
        self.parameters['U'], _ = np.linalg.qr(tmp,mode='reduced')

    def transform(self,points):
        return np.matmul(points,self.parameters['U'])

    def loss(self,points):
        points = points - self.hyperparameters['mean_center']*np.expand_dims(self.parameters['mean'],axis=0)
        utx = np.matmul(points,self.parameters['U'])
        residuals = points - np.matmul(utx,self.parameters['U'].T)
        loss = np.linalg.norm(residuals)**2
        return loss
