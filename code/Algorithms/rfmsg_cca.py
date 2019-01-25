from msg_pls import MSG_PLS
import numpy as np

class RFMSG_CCA(MSG_PLS,object):
    def __init__(self,hyperparameters):
        super(RFMSG_CCA, self).__init__(hyperparameters)
        self.hyperparameters=hyperparameters
        assert 'mx' in hyperparameters, 'number of random features for view 1 not specified'
        assert 'my' in hyperparameters, 'number of random features for view 1 not specified'
        assert self.hyperparameters['mx'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of the feature space'
        assert self.hyperparameters['my'] >= self.hyperparameters['k'], \
        'dimensionality of projection must be smaller than that of the feature space'
        self.parameters={
                'M': np.zeros((self.hyperparameters['mx'], self.hyperparameters['my'])),
                'M_bar': np.zeros((self.hyperparameters['mx'], self.hyperparameters['my'])),
                'mean_x': np.zeros(self.hyperparameters['mx']),
                'mean_y': np.zeros(self.hyperparameters['my']),
                'm': min(self.hyperparameters['mx'], self.hyperparameters['my']),
                't': 0,
                'rfSamples_x': self.randomFeatureSamples(hyperparameters['kernel'],VIEW='x'),
                'rfSamples_y': self.randomFeatureSamples(hyperparameters['kernel'],VIEW='y')
                }

    def step(self,point):
        x,y=point
        rf_x = np.array(self.randomFeature(x,VIEW='x'))
        rf_y = np.array(self.randomFeature(y,VIEW='y'))
        super(RFMSG_CCA, self).step([rf_x,rf_y])
        print(super(RFMSG_CCA, self).loss([rf_x,rf_y]))


    def randomFeatureSamples(self,kernel,VIEW):
        if VIEW=='x':
            m = self.hyperparameters['mx']
            d = self.hyperparameters['dx']
        else:
            m = self.hyperparameters['my']
            d = self.hyperparameters['dy']
        if self.hyperparameters['kernel'] =='rbf':
            mean = np.zeros(d)
            cov = self.hyperparameters['kernel_hyperparameter']*np.eye(d)
            W = np.random.multivariate_normal(mean, cov, m)
            B = np.random.uniform(0,2*np.pi,m)
            return W,B

    def randomFeature(self,point,VIEW):
        if VIEW=='x':
            m = self.hyperparameters['mx']
            rfSamples = self.parameters['rfSamples_x']
        else:
            m = self.hyperparameters['my']
            rfSamples = self.parameters['rfSamples_y']
        if self.hyperparameters['kernel'] == 'rbf':
            W,B = rfSamples
            const= np.sqrt(2)/np.sqrt(m)
            rf_point = [const*np.cos(np.dot(point,w)+b) for (w,b) in zip(W,B)]
            return rf_point







