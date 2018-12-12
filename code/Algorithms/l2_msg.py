from msg import MSG
import numpy as np

class l2MSG(MSG,object):
    def __init__(self,hyperparameters):
        super(l2MSG, self).__init__(hyperparameters)
        self.hyperparameters=hyperparameters
        assert 'lambda' in hyperparameters, 'regularization parameter not specified'

    def step(self,point):

        super(l2MSG, self).step(point,IF_PROJECT=0)
        tmp = self.parameters['P'] - self.hyperparameters['lambda']*self.hyperparameters['learning_rate']*self.parameters['P']
        [eigenValues,eigenVectors]= self.projection_fast(tmp)
        self.parameters['P'] = self.rounding(eigenValues,eigenVectors)
        print self.loss(point)
