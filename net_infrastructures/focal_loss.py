from math import log
import operator
def createDataset():
    dataset=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no'],]
    labels=['no surfacing','flippers']
    return dataset,labels

X_train,y_train=createDataset()
labels=y_train.copy()


