import sys
import os
import numpy as np
from pickle import pkl
from data_streamer import Streamer

# TODO import all the algorithms we wrote

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
assert os.path.isfile(sys.argv[1]), \
        'first and only argument of this script is a config file'
config=parse_config(sys.argv[1])
if len(config['data'].split(','))>1:
    stream = MultiStreamer(config['data'].split(',')).get_stream()
else:
    stream = Streamer(config['data']).get_stream()
# TODO call the correct algorithm
algorithm=something(config['algorithm'])
for point in stream:
    algorithm.step(point)
if not config['savefile'].endswith('.pkl'):
    config['savefile'] += '.pkl'
print('saving to ' + config['savefile'] + '...')
with open(config['savefile'], 'wb') as f:
    pkl.dump(algorithm,f)
