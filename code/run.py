import sys
import numpy as np
from data_streamer import Streamer

""" BEGIN UTIL FUNCTIONS """
def interpret(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val
def parse_config(config_file_name):
    config={}
    prefix=''
    for line in open(config_file_name,'r'):
        content=line.strip('\n').split('#')[0].strip(' ')
        if len(content)==0:
            continue
        if (content[0]=='[') and (content[-1]==']'):
            upper_key=content[1:-1]
            assert upper_key not in config, "repeated entry in config file"
            config[upper_key]={}
            continue
        lower_key=content.split('=')[0].strip(' ')
        assert lower_key not in config[upper_key], "repeated settings in config"
        value=content.split('=')[1].strip(' ')
        config[upper_key][lower_key]=value
    return config
""" END UTIL FUNCTIONS """

args = parser.parse_args()
assert args.n_components <= args.dimensionality
stream = Streamer(args.data).get_stream()
U = np.eye(args.dimensionality, args.n_components)
mean = np.zeros(args.dimensionality)
time = 0
for point in stream:
    time += 1
    learning_rate = 1./math.sqrt(time)
    point=np.array(point,dtype=np.float32)
    mean = ((time - 1.0)*mean + point)/time # update running average
    point -= mean # center point
    V = U + learning_rate * np.matmul(np.outer(point,point),U)
    U, _ = np.linalg.qr(V,mode='reduced')
print('saving to ' + args.savefile + '...')
np.save(args.savefile, U)
