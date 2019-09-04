import argparse
import random
from random import randint

'''
to filter the sub set from train-set, in order to get sub-set for evaluation trainded model's.
if the trained model has been coveraged at train-set. 
while coveraged at train-set is the basic goal, before focusing on the over-fitting.

python random_filter.py --inputfile trainval.txt
'''
if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--inputfile', type=str, help='input file for filter ', required=True)
    _args = parse.parse_args()
    input_filter_file = _args.inputfile;
    resultList = random.sample(range(0, 23125), 5000)
    train_subset = open('train_subset.txt','w')
    with open(input_filter_file,'r') as f:
        lines=f.readlines()
        for index in resultList:
            train_subset.write(lines[index])
    
    train_subset.close()
    
