from msg import MSG
import numpy as np

class minibatchMSG(MSG,object):
    def __init__(self,hyperparameters):
        super(minibatchMSG, self).__init__(hyperparameters)
        self.hyperparameters=hyperparameters
        assert 'm' in hyperparameters, 'minibatch size not specified'
        self.counter = 0

    def step(self,point):
        self.counter+=1
        if self.counter == self.hyperparameters['m']:
            super(minibatchMSG, self).step(point,IF_PROJECT=1)
            self.counter=0
        else:
            super(minibatchMSG, self).step(point,IF_PROJECT=0)

        print super(minibatchMSG, self).loss(point)


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
