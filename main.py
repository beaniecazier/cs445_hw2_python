#!/usr/bin/python

import sys
import pandas as pd
import numpy as np

#csv to df
trainingset = pd.read_csv("", header=None)
testset = pd.read_csv("", header=None)
#extract the first column as targets
#replace first column with 255
#preprocess by dividing by 255
train_data.drop(columns=0)     # Remove column with target info

def LoadData():

def RunSimulation():

if __name__ == "__main__":
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    pass



    


EPOCH_MAX  = 2
INPUT_MAX  = 255
rate = 1

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE  = "../mnist_test.csv"
output = [None]*10 # used to store outputs and evaluate the prediction of network
accuracy = pandas.DataFrame(0,index=range(0,EPOCH_MAX+1),columns=['test_c','test_i','train_c','train_i'])

# set up confusion matrix: rows=actual, col=predicted
confu  = pandas.DataFrame(0,index=range(0,10),columns=range(0,10)) 

# Import data
train_data = pandas.read_csv(TRAIN_FILE,header=None)
test_data  = pandas.read_csv(TEST_FILE ,header=None)

# Preprocess data 
train_data.sample(frac=1)      # shuffle training data
train_target = train_data[0].values # Save targets as a separate dataframe/array
train_data.drop(columns=0)     # Remove column with target info
train_data = train_data.values # convert to numpy array
train_data = numpy.divide(train_data, INPUT_MAX) # scale inputs between 0 and 1 by dividing by input max value

test_target = test_data[0].values # Save targets as a separate dataframe/array
test_data.drop(columns=0)    # Remove column with target info
test_data = test_data.values # convert to numpy array
test_data = numpy.divide(test_data, INPUT_MAX)

input_size = len(train_data[0]) # how many inputs are there