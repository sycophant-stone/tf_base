
import numpy as np
import os
import argparse

def calc_mean(inputfile):
    timeval=[]
    with open(inputfile, 'r') as f:
        for line in f.readlines():
            if "Running time" in line and "Couldn't find any detections" not in line:
                words=line.split(' ')
                timeval.append(float(words[-2]))
    narray = np.array(timeval)
    print(np.mean(narray))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--input', type=str, help='input log with time-specific log lines')
    _args = parse.parse_args()
    input_file = _args.input;


    calc_mean(inputfile=input_file)
