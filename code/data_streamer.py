import sys
import numpy as np

class Streamer:
    def __init__(self,filename):
        self.filename=filename
        self.stream=None

    def get_stream(self):
        self.stream=open(self.filename,'r')
        line=self.stream.readline()
        while line != '':
            point=np.array([float(x) for x in line[:-1].split(',')])
            yield tuple(point)
            line=self.stream.readline()
