""" Matrix Stochastic Gradient method for Canonical Correlation Analysis as described in 
Raman Arora, Poorya Mianjy, and Teodor Marinov. Stochastic optimization for multiview representation learning using partial least squares, 2016."""

import numpy as np

class MSG_CCA:
    def __init__(self,hyperparameters):
        self.hyperparameters=hyperparameters
        assert 'dx' in hyperparameters, 'dimensionality of first view not specified'
        assert 'dy' in hyperparameters, 'dimensionality of second view not specified'
        assert 'k' in hyperparameters, 'dimensionality of projection not specified'
        assert 'learning_rate' in hyperparameters, \
                'initial learning rate not specified'
        assert self.hyperparameters['dx'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of original space'
        assert self.hyperparameters['dy'] >= self.hyperparameters['k'], \
                'dimensionality of projection must be smaller than that of original space'
        self.hyperparameters.setdefault('mean_center',1.0)
        assert self.hyperparameters['mean_center'] in [0.0, 1.0], \
                "either mean center or don't"
        self.parameters={
                #'M': np.zeros((self.hyperparameters['dx'], self.hyperparameters['dy'])),
                'M': np.random.uniform( low=-1.0, high=1.0, \
                        size=(self.hyperparameters['dx'], self.hyperparameters['dy']) ),
                'M_bar': np.zeros((self.hyperparameters['dx'], self.hyperparameters['dy'])),
                'mean_x': np.zeros(self.hyperparameters['dx']),
                'mean_y': np.zeros(self.hyperparameters['dy']),
                'd': min(self.hyperparameters['dx'], self.hyperparameters['dy']),
                't': 0
                }

    def step(self,point):
        x,y=point
        self.parameters['t'] += 1
        # update running averages
        alpha = 1.0/self.parameters['t']
        self.parameters['mean_x'] = (1-alpha)*self.parameters['mean_x'] + alpha*x
        self.parameters['mean_y'] = (1-alpha)*self.parameters['mean_y'] + alpha*y
        x -= self.hyperparameters['mean_center']*self.parameters['mean_x']
        y -= self.hyperparameters['mean_center']*self.parameters['mean_y']
        gradient = np.outer(x,y)
        tmp = self.parameters['M'] + self.hyperparameters['learning_rate'] * gradient
        self.parameters['M'] = self.projection(tmp)
        self.parameters['M_bar'] = (1-alpha)*self.parameters['M_bar'] \
                + alpha*self.parameters['M']
        # self.rounding() not necessary to do this at every time step

    def find_S(self,sigma,kappa):
        """ finding S such that
        sigma(proj)_i = max(0, min(1, sigma(mat)_i + S)) where S is chosen such that
        \sum_i sigma(proj)_i = k
        Algorithm 2 in Raman Arora, Andy Cotter, and Nati Srebro. Stochastic optimization 
        of pca with capped msg, 2013."""
        # in the original paper, indexing is 1-based, hence the -1 to switch to
        # Python's 0-based indexing
        ii,jj,si,sj,ci,cj=1,1,0,0,0,0
        n=len(sigma)
        while (ii <= n):
            if ( ii < jj ):
                S=( self.hyperparameters['k'] - (sj-si) - (self.parameters['d']-cj) )\
                        /(cj-ci)
                b=[sigma[ii-1]+S >= 0, sigma[jj-2]+S <= 1,
                        (ii<=1) or sigma[ii-2]+S <=0, (jj>=n) or (sigma[jj]>=1)]
                b=all(b)
                if b:
                    return S
            #end if ( ii<jj )
            if ((jj<=n) and (sigma[jj-1] - sigma[ii-1] <= 1)):
                sj += kappa[sigma[jj-1]]*sigma[jj-1]
                cj += kappa[sigma[jj-1]]
                jj += 1
            else:
                si += kappa[sigma[ii-1]]*sigma[ii-1]
                ci += kappa[sigma[ii-1]]
                ii += 1
            #end if ((jj<=n) and (sigma[jj-1] - sigma[ii-1] <= 1))
        # end while ( ii <= n )
        assert False, "error occurred"

    def projection(self,mat):
        """ projection of a matrix onto the feasible set
        the projection proj has the same singular vectors of mat, and its singular values are
        sigma(proj)_i = max(0, min(1, sigma(mat)_i + S)) where S is chosen such that
        \sum_i sigma(proj)_i = k """
        U,S,VT = np.linalg.svd(mat)
        sigma=sorted(S.tolist()) # we want the eigenvalues to be in ascending order
        kappa={} # multiplicities
        for s in sigma:
            kappa.setdefault(s,0)
            kappa[s]+=1
        sigma=sorted(list(set(sigma))) # remove duplicates
        s=self.find_S(sigma,kappa)
        new_sigma=np.diag([max(0, min(1, sig+s)) for sig in S])
        new_S=np.zeros((U.shape[0],VT.shape[0]))
        np.fill_diagonal(new_S, new_sigma) # modify new_S in place
        return np.matmul(np.matmul(U,new_S),VT)

    def rounding(self):
        """ implement rounding operation mentioned in Algorithm 1, line 6 """
        U,S,VT=np.linalg.svd(self.parameters['M_bar'])
        assert (sorted(S)==S[::-1]).all() # make sure it's sorted in descending order
        S=[s if ii<=self.hyperparameters['k'] else 0. for ii,s in enumerate(S)]
        diag_S=np.zeros((U.shape[0],VT.shape[0]))
        np.fill_diagonal(diag_S, S) # modify diag_S in place
        self.parameters['M_tilde']=np.matmul(np.matmul(U,diag_S),VT)

    def transform(self,points):
        x,y=points
        self.rounding() # update M_tilde
        assert False, "not implemented"

    def loss(self,points):
        x,y=points
        self.rounding() # update M_tilde
        x -= self.hyperparameters['mean_center']*self.parameters['mean_x']
        y -= self.hyperparameters['mean_center']*self.parameters['mean_y']
        loss=np.trace(np.matmul(np.matmul(x,self.parameters['M_tilde']),y.T))
        return loss
