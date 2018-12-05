import Oja

class RFOja(Oja):
    def __init__(self,hyperparameters):
        self.hyperparameters=hyperparameters
        assert 'd' in hyperparameters, 'dimensionality of input vectors not specified'
        assert 'k' in hyperparameters, 'dimensionality of projection not specified'
        assert 'm' in hyperparameters, 'number of random features not specified'
        assert 'learning_rate' in hyperparameters, \
                'initial learning rate not specified'
        assert self.hyperparameters['m'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of the feature space'
        self.parameters={
                'U': np.eye(self.hyperparameters['m'], self.hyperparameters['k']),
                'mean': np.zeros(self.hyperparameters['m']),
                't': 0
                }

    def step(self,point):
        self.parameters['t'] += 1
        step_size = self.hyperparameters['learning_rate']**0.5
        # update running average
        alpha = 1.0/self.parameters['t']
        self.parameters['mean'] = \
                (1-alpha)*self.parameters['mean'] \
                + alpha*point
        point -= self.parameters['mean']
        gradient = np.matmul(np.outer(point,point),self.parameters['U'])
        tmp = self.parameters['U'] + step_size * gradient
        self.parameters['U'], _ = np.linalg.qr(tmp,mode='reduced')

    def transform(self,points):
        return np.matmul(points,self.parameters['U'])

    def loss(self,points):
        uut = np.matmul(self.parameters['U'],self.parameters['U'].T)
        residuals = points - np.matmul(points,uut)
        loss = np.sum(np.power(residuals,2))
        return loss