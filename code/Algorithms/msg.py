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

    def step(self,point,IF_PROJECT=1):
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
        if IF_PROJECT:
            [eigenValues,eigenVectors]= self.projection_fast(tmp)
            self.parameters['P'] = self.rounding(eigenValues,eigenVectors)
        else:
            self.parameters['P'] = tmp
        # print self.loss(point)

    def transform(self,points):
        return np.matmul(self.parameters['P'], points)

    def loss(self,points):
        residuals = points - np.matmul(self.parameters['P'], points)
        loss = np.sum(np.power(residuals,2))
        return loss

    


    def projection_slow(self,P):
   
        d = self.hyperparameters['d']
        k = self.hyperparameters['k']
    
        # Compute eigenvectors and eigenvalues
        eigenValues, eigenVectors = np.linalg.eig(P)

        idx = eigenValues.argsort()[::1] 
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]


        # Copy eigenvalues to a new vector
        eigenValues_copy = eigenValues.copy()

        # Set values less than 0 to 0 and greater than 1 to 1
        for i in range(d):
            if eigenValues_copy[i] < 0:
                eigenValues_copy[i]=0
            elif eigenValues_copy[i]>1:
                eigenValues_copy[i]=1

        if sum(eigenValues_copy) <=k:
            return [eigenValues_copy,eigenVectors]

        # trace was bigger than k after capping ==> shf <= 0
        # rule: SS(i:j) should not be capped and everything outside that range
        # should be capped accordingle, i.e. SS(1:i-1)=0 and SS(j+1:l)=1
        l=len(eigenValues)
        for i in range(l):
            # print i
            for j in range(l):
                eigenValues_copy = eigenValues.copy()
                if i>0:
                    eigenValues_copy[0:i]=0
                if j<l-1:
                   eigenValues_copy[j+1:l]=1
                # print i,j
                shf = 0 if (j-i+1) == 0 else (k-sum(eigenValues_copy))/(j-i+1)  
                eigenValues_copy[i:j+1]=eigenValues_copy[i:j+1]+shf
                 # check for consistency
                if (eigenValues_copy[i]>=0  and (i==0 or eigenValues[i-1]+shf<=0)  and eigenValues_copy[j]<=1  and (j==l-1 or eigenValues[j+1]+shf>=1)):
                    # print eigenValues_copy
                    return [eigenValues_copy,eigenVectors]

    def projection_fast(self,P):

        d = self.hyperparameters['d']
        k = self.hyperparameters['k']
        projectedP = np.zeros([d,d])
       # Compute eigenvectors and eigenvalues
        eigenValues, eigenVectors = np.linalg.eig(P)


        idx = eigenValues.argsort()[::1] 
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        # print eigenValues

        # Copy eigenvalues to a new vector
        eigenValues_copy = eigenValues.copy()

        # Set values less than 0 to 0 and greater than 1 to 1
        for i in range(d):
            if eigenValues_copy[i] < 0:
                eigenValues_copy[i]=0
                eigenValues[i]=0
            elif eigenValues_copy[i]>1:
                eigenValues_copy[i]=1


        if sum(eigenValues_copy) <=k:
            return[eigenValues,eigenVectors]

        eigenValues_copy = eigenValues.copy()

        i=0
        j=0

        while 1-eigenValues_copy[j] > 0 and j < d-1:
            j = j+1

        while j<d-2:
            # print j,d
            s_ij = 1-eigenValues_copy[j+1]
            while (eigenValues_copy[i] + s_ij <= 0) and (i <= j):
                i = i+1

            if i>j:
                j = j+1
            eigenValues_add = [q+s_ij for q in  eigenValues_copy]
            eigenValues_add = [min([1,x]) for x in eigenValues_add]
            eigenValues_add = [max([0,x]) for x in eigenValues_add]

            p_ij = sum(eigenValues_add)
            if p_ij > k:
                j = j+1
                continue
            if p_ij == k:
                return [eigenValues_add, eigenVectors]

            i = 0
            while i<=j:
                s_ij = (k-(d-j+1)-sum(eigenValues[i:j+1]))/(j-i+1)
                if s_ij <0 and eigenValues_copy[i] + s_ij >= 0 and ((i > 0 and eigenValues[i-1] + s_ij <= 0) or i==0) and eigenValues_copy[j] + s_ij <= 1 and eigenValues[j+1] + s_ij >= 1:
                    eigenValues_add = [q+s_ij for q in  eigenValues_copy]
                    eigenValues_add = [min([1,x]) for x in eigenValues_add]
                    eigenValues = [max([0,x]) for x in eigenValues_add]

                    return [eigenValues, eigenVectors]
                i = i+1

        for j in range(d-2,d):
            i = 0
            while i<=j:
                s_ij = (k-(d-j-1)-sum(eigenValues[i:j+1]))/(j-i+1)
                if s_ij < 0 and eigenValues_copy[i] + s_ij >= 0 and ((i > 0 and eigenValues[i-1] + s_ij <= 0) or i==0) and eigenValues_copy[j] + s_ij <= 1 and ((j<(d-1) and eigenValues[j+1] + s_ij >= 1) or j==(d-1)):
                    eigenValues_add = [q+s_ij for q in  eigenValues_copy]
                    eigenValues_add = [min([1,x]) for x in eigenValues_add]
                    eigenValues = [max([0,x]) for x in eigenValues_add]

                    return [eigenValues, eigenVectors]
                i = i+1
        eigenValues[0:d-k] = 0
        eigenValues[d-k:d] = 1

        return [eigenValues_add, eigenVectors]


    def rounding(self,eigenValues,eigenVectors):
        d = self.hyperparameters['d']
        k = self.hyperparameters['k']

        projectedP = np.matmul(eigenVectors[:,d-k-1:d],np.matmul(np.diag(eigenValues[d-k-1:d]),eigenVectors[:,d-k-1:d].T))
        # for q in range(d-1,d-k-1,-1):
        #     projectedP = projectedP + (eigenValues[q]*np.outer(eigenVectors[q],eigenVectors[q]))
        return projectedP



