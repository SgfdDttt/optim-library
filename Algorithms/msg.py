""" Matrix Stochastic Gradient Descnet (MSG) as described in ... ADD REFERENCE """
import numpy as np

class MSG:
    def __init__(self,hyperparameters):
        self.hyperparameters=hyperparameters
        assert 'd' in hyperparameters, 'dimensionality of input vectors not specified'
        assert 'k' in hyperparameters, 'dimensionality of projection not specified'
        assert 'learning_rate' in hyperparameters, \
                'initial learning rate not specified'
        assert self.hyperparameters['d'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of original space'
        self.parameters={
                'P': np.eye(self.hyperparameters['d'], self.hyperparameters['d']),
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
        gradient = np.outer(point,point)
        tmp = self.parameters['P'] + step_size * gradient
        self.parameters['P'] = self.projection(tmp)

    def transform(self,points):
        return np.matmul(self.parameters['P'], points)

    def loss(self,points):
        residuals = points - np.matmul(self.parameters['U'], points)
        loss = np.sum(np.power(residuals,2))
        return loss

    def projection(self):
   
		eigenValues, eigenVectors = linalg.eig(A)
		idx = eigenValues.argsort()[::1] 
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]
		projectedP = np.array(self.hyperparameters['d'],self.hyperparameters['d'])
		projectedP = [projectedP + eigenValues[i]np.outer(eigenVectors[i],eigenVectors[i]) for i in range(self.hyperparameters['k'])]

		return projectedP
  #   	eigenValues, eigenVectors = np.linalg.eig(A)

		# idx = eigenValues.argsort()[::-1]   
		# eigenValues = eigenValues[idx]
		# eigenVectors = eigenVectors[:,idx]


