from oja import Oja
import numpy as np

class RFOja(Oja,object):
    def __init__(self,hyperparameters):
        super(RFOja, self).__init__(hyperparameters)
        self.hyperparameters=hyperparameters
        assert 'm' in hyperparameters, 'number of random features not specified'
        assert self.hyperparameters['m'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of the feature space'
        self.parameters={
                'U': np.eye(self.hyperparameters['m'], self.hyperparameters['k']),
                'mean': np.zeros(self.hyperparameters['m']),
                't': 0,
                'rfSamples': self.randomFeatureSamples(hyperparameters['kernel'])
                }

    def step(self,point):
        rf_point = np.array(self.randomFeature(point))
        super(RFOja, self).step(rf_point)
        print(super(RFOja, self).loss(rf_point))


    def randomFeatureSamples(self,kernel):
        if self.hyperparameters['kernel'] =='rbf':
            mean = np.zeros(self.hyperparameters['d'])
            cov = self.hyperparameters['kernel_hyperparameter']*np.eye(self.hyperparameters['d'])
            W = np.random.multivariate_normal(mean, cov, self.hyperparameters['m'])
            B = np.random.uniform(0,2*np.pi,self.hyperparameters['m'])
            return W,B

    def randomFeature(self,point):
        if self.hyperparameters['kernel'] == 'rbf':
            W,B = self.parameters['rfSamples']
            const= np.sqrt(2)/np.sqrt(self.hyperparameters['m'])
            rf_point = [const*np.cos(np.dot(point,w)+b) for (w,b) in zip(W,B)]
            return rf_point
