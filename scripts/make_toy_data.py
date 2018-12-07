""" copied from https://bitbucket.org/adrianbenton/dgcca-py3/src/master/ """
import numpy as np
import sys
import math
import torch

def make_data():
    ### Generate sample data with 3 views and two classes:
    ### 1) Two concentric circles
    ### 2) Two parabolas with different intercept and quadratic coefficient
    ### 3) Two concentric circles with classes reversed and greater variance
    N = 1000000 # Number of examples
    F1 = 2 # Number of features in view 1
    F2 = 2
    F3 = 2
    k = 2 # Number of latent features

    # First half of points belong to class 1, second to class 2
    G = np.zeros( ( N, k ) )

    G[:int(N/2),0] = 1.0
    G[int(N/2):,1] = 1.0
    classes = ['Class1' for i in range(int(N/2))] + ['Class2' for i in range(int(N/2))]

    # Each class lies on a different concentric circle
    rand_angle = np.random.uniform(0.0, 2.0 * math.pi, (N, 1) )
    rand_noise = 0.1 * np.random.randn(N, k)
    circle_pos = np.hstack( [np.cos(rand_angle), np.sin(rand_angle)])
    radius = G.dot(np.asarray( [[1.0], [2.0]] )).reshape( (N, 1) )
    V1 = np.hstack([radius, radius]) * circle_pos + rand_noise

    # Each class lies on a different parabola
    rand_x = np.random.uniform(-3.0, 3.0, (N, 1) )
    rand_noise = 0.1 * np.random.randn(N, k)
    intercepts = G.dot( np.asarray([[0.0], [1.0]])).reshape( (N, 1) )
    quadTerms = G.dot( np.asarray( [[2.0], [0.5]] )).reshape( (N, 1) )
    V2 = np.hstack( [rand_x, intercepts + quadTerms * (rand_x*rand_x)]) + rand_noise

    # Class on inside is drawn from a gaussian, class on outside is on a concentric circle
    rand_angle = np.random.uniform(0.0, 2.0 * math.pi, (N, 1) )
    inner_rand_noise = 1.0 * np.random.randn(int(N/2), k) # More variance on inside
    outer_rand_noise = 0.1 * np.random.randn(int(N/2), k)
    rand_noise = np.vstack( [outer_rand_noise, inner_rand_noise] )
    circle_pos = np.hstack( [np.cos(rand_angle), np.sin(rand_angle)])
    radius = G.dot(np.asarray( [[2.0], [0.0]] )).reshape( (N, 1) )
    V3 = np.hstack([radius, radius]) * circle_pos + rand_noise

    # We have no missing data
    K = np.ones( (N, 3) )

    # Gather into dataframes to plot
    views = [V1, V2, V3]
    return views

def save_to_csv(matrix,filename):
    """ matrix is a list of lists """
    with open(filename,'w') as f:
        f.write('\n'.join(','.join(str(x) for x in v) for v in matrix))
        f.write('\n')

data=make_data()
data=[x.tolist() for x in data]
root_name='toy_data_view'
file_names=[root_name+str(ii+1)+'.csv' for ii in range(3)]
for d,fname in zip(data,file_names):
    save_to_csv(d,fname)
