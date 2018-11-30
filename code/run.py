import sys
import os
import numpy as np
import pickle as pkl
import Algorithms
from data_streamer import Streamer, MultiStreamer

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
        key=content.split('=')[0].strip(' ')
        assert key not in config, "repeated settings in config"
        value=content.split('=')[1].strip(' ')
        config[key]=interpret(value)
    return config
""" END UTIL FUNCTIONS """
assert os.path.isfile(sys.argv[1]), \
        'first and only argument of this script is a config file'
print('config file: ' + str(sys.argv[1]))
config=parse_config(sys.argv[1])
print('config parameters: ' + str(config))
if len(config['data'].split(','))>1:
    stream = MultiStreamer(config['data'].split(',')).get_stream()
else:
    stream = Streamer(config['data']).get_stream()
# pass full config as hyperparameters
algorithm=getattr(Algorithms,config['algorithm'])(config)
for point in stream:
    algorithm.step(point)
if not config['savefile'].endswith('.pkl'):
    config['savefile'] += '.pkl'
print('saving to ' + config['savefile'] + '...')
with open(config['savefile'], 'wb') as f:
    pkl.dump(algorithm,f)
