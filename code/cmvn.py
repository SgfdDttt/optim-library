import sys
import numpy as np
""" input is a .csv file
output (to stdout) is the data contained in the input file, but mean-centered
and variance-normalized
example usage: python cmvn.py mnist.csv > mnist_normalized.csv """
cmvn={'N':0}
for line in open(sys.argv[1],'r'):
    point=np.array([float(x) for x in line.strip('\n').split(',')])
    cmvn.setdefault('mean',np.zeros_like(point))
    cmvn.setdefault('variance',np.zeros_like(point))
    cmvn['N']+=1
    alpha=1.0/cmvn['N']
    cmvn['mean']=(1-alpha)*cmvn['mean']+alpha*point
    cmvn['variance']=(1-alpha)*cmvn['variance']+alpha*np.power(point,2)
cmvn['variance']=np.sqrt( cmvn['variance'] - np.power(cmvn['mean'],2) )
# if some part of variance is 0, set it to 1 instead
cmvn['variance']=np.where(cmvn['variance']==0.0,1.0,cmvn['variance'])

for line in open(sys.argv[1],'r'):
    point=np.array([float(x) for x in line.strip('\n').split(',')])
    npoint=np.divide(point-cmvn['mean'],cmvn['variance'])
    print(','.join([ str(x) for x in npoint.tolist() ]))
