#!/usr/bin/python

import sys
import neural_net

import numpy as np
import pandas as pd

test_file = ''
train_file = ''
confmat_file = ''
accarr_file = ''

# #csv to df
trainingset = pd.read_csv("test.csv", header=None)
# testset = pd.read_csv("", header=None)
# #extract the first column as targets
# #replace first column with 255
# #preprocess by dividing by 255
# train_data.drop(columns=0)     # Remove column with target info

# def LoadData():
#     return

# def RunSimulation():
#     return

#argument options are 
# -i for input training csv file name
# -t for test csv set
# -a for output csv accuracy file
# -o for output file name
# -m for num of processes to use
# -h for csv file with hyperparameters(has header line)
if __name__ == "__main__":
	#print 'Number of arguments:', len(sys.argv), 'arguments.'
	#print 'Argument List:', str(sys.argv)
	for arg in sys.argv:
		if arg == 'HELP':
			Help()
			sys.exit()

	try:
		opts, args = getopt.getopt(argv, "hiotam")
	except:
		Help()
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			SetHyperparamters(pd.read_csv(arg))
		elif opt == '-i':
		elif opt == '-o':
		elif opt == '-t':
		elif opt == '-a':
		elif opt == '-m':
		
	pass


TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "mnist_test.csv"

# set up accuracy recording
accuracy = pd.DataFrame(0.0, index=range(
	0, EPOCH_MAX+1), columns=['test', 'train'])

# set up confusion matrix: rows=actual, col=predicted
confmat_train = pd.DataFrame(0, index=range(0, 10), columns=range(0, 10))
confmat_test = pd.DataFrame(0, index=range(0, 10), columns=range(0, 10))

# load data
train_data = pd.read_csv(TRAIN_FILE, header=None)
test_data = pd.read_csv(TEST_FILE, header=None).values
# # Preprocess data 
train_data.sample(frac=1)      # shuffle training data
if VERBOSE:
    print('now randomizing the training data and separating out the targets')

# Save targets as a separate dataframe/array
target_train = train_data[:, 0]
train_data = np.array(train_data)
train_data[:, 0] = INPUT_MAX
if VERBOSE:
    print('training data targets:')
    print(target_train)
    print('training data:')
    print(train_data)
    print('the shape of this set of data is:')
    print(train_data.shape)
    print('now separating out the targets from the testing data')
target_test = test_data[:, 0]
test_data = np.array(test_data)
test_data[:, 0] = INPUT_MAX
if VERBOSE:
    print('testing data targets:')
    print(target_test)
    print('testing data:')
    print(test_data)
    print('the shape of this set of data is:')
    print(test_data.shape)

input_size = len(train_data[0])  # how many inputs are there
if VERBOSE:
    print('the number of inputs is:', input_size)

# Preprocess data
train_data = np.divide(train_data, INPUT_MAX)
if VERBOSE:
    print('training data after preprocessing')
    print(train_data)
test_data = np.divide(test_data, INPUT_MAX)
if VERBOSE:
    print('testing data after preprocessing')
    print(test_data)

# initialize neural network


# Generate the final confusion matrices and print the data
confmat_test = network.confusionmatrix(network.predict(test_data), target_test)
confmat_train = network.confusionmatrix(network.predict(train_data), target_train)
accuracy['test'].to_csv('accuracy_rate_test'+str(LRATE)+'.csv')
accuracy['train'].to_csv('accuracy_rate_train'+str(LRATE)+'.csv')
pd.DataFrame(confmat_train).to_csv('confusionmatrix_train'+str(LRATE)+'.csv')
pd.DataFrame(confmat_test).to_csv('confusionmatrix_test'+str(LRATE)+'.csv')
