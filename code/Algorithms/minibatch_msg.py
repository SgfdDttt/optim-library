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
        print(self.counter)
        if self.counter == self.hyperparameters['m']:
            super(minibatchMSG, self).step(point,IF_PROJECT=1)
            self.counter=0
        else:
            super(minibatchMSG, self).step(point,IF_PROJECT=0)

        # print super(minibatchMSG, self).loss(point)
