
import numpy as np
import os

def calc_mean(inputfile):
    timeval=[]
    with open(inputfile, 'r') as f:
        for line in f.readlines():
            if "Running time" in line and "Couldn't find any detections" not in line:
                words=line.split(' ')
                timeval.append(float(words[2]))
    narray = np.array(timeval)
    print(np.mean(narray))

if __name__ == '__main__':
    calc_mean(inputfile="5cpu72.log")
