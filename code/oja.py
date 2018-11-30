import argparse
import math
import numpy as np
from data_streamer import Streamer
from data_streamer import MultiStreamer

parser = argparse.ArgumentParser(description="Run Oja's algorithm using Numpy")
parser.add_argument('--data', '-s', type=str,
                    help='file to be used as data stream')
parser.add_argument('--n_components', '-k', type=int,
                    help='number of components of CCA decomposition')
parser.add_argument('--dimensionality', '-d', type=int,
                    help='dimensionality of input data')
parser.add_argument('--savefile', type=str,
                    help='file to save final transformation to')
args = parser.parse_args()
assert args.n_components <= args.dimensionality
stream = Streamer(args.data).get_stream()
stream = MultiStreamer([args.data,args.data]).get_stream()
U = np.eye(args.dimensionality, args.n_components)
mean = np.zeros(args.dimensionality)
time = 0
for point in stream:
    print(point)
    print(point[0])
    assert False
    time += 1
    learning_rate = 1./math.sqrt(time)
    point=np.array(point,dtype=np.float32)
    mean = ((time - 1.0)*mean + point)/time # update running average
    point -= mean # center point
    V = U + learning_rate * np.matmul(np.outer(point,point),U)
    U, _ = np.linalg.qr(V,mode='reduced')
print('saving to ' + args.savefile + '...')
np.save(args.savefile, U)
