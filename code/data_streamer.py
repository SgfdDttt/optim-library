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
            yield point
            line=self.stream.readline()

class MultiStreamer:
    def __init__(self,filenames):
        self.filenames=list(filenames)
        self.stream=None

    def get_stream(self):
        self.streams=[open(f,'r') for f in self.filenames]
        lines=[s.readline() for s in self.streams]
        while lines[0] != '':
            points=[np.array([float(x) for x in line[:-1].split(',')]) \
                    for line in lines]
            yield tuple(points)
            lines=[s.readline() for s in self.streams]
