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
                'rfSamples': self.randomFeatureSamples()                 
                }
        self.hyperparameters['KERNEL_MATRIX_IN_MEMORY'] = 0

    def step(self,point):
        rf_point = np.array(self.randomFeature(point))
        super(RFOja, self).step(rf_point)
        # print super(RFOja, self).loss(rf_point)


    def randomFeatureSamples(self):
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

    def kernelMatrix(points):
        n_points = len(points)
        K = np.zeros(n_points)
        if self.hyperparameters['kernel'] =='rbf':
            for point1 in points:
                for point2 in points:
                    K[i][j] = np.exp(-np.linalg.norm(point1-point2)**2/self.hyperparameters['kernel_hyperparameter'])

    def loss(self,points):
        if self.hyperparameters['kernel'] =='rbf':
            rf_points = np.array([self.randomFeature(q) for q in points])
            n_points = len(rf_points)
            if not self.hyperparameters['KERNEL_MATRIX_IN_MEMORY']:
                C_m = np.outer(rf_points,rf_points.T)/n_points
                eVEc,_ = np.eigh(C_m)
                self.hyperparameters['eVec_k']=eVec[:k]
                self.hyperparameters['K'] = kernelMatrix(points, kernel)
                self.hyperparameters['KERNEL_MATRIX_IN_MEMORY'] = 1

            S = [1/(np.dot(q,np.multiply(C_m,q))**(0.5)) for q in self.parameters['U']]


            V = np.muliply(self.hyperparameters['eVec_k'].T, np.multiply(self.parameters['U'],diag(S)))
            loss = np.trace(np.multiply(V.T,self.hyperparameters['K']),V)
            # super(RFOja, self).loss(rf_points)
            # utx = np.matmul(rf_points,self.parameters['U'])
            # residuals = rf_points - np.matmul(utx,self.parameters['U'].T)
            # loss = np.linalg.norm(residuals)**2
            return loss