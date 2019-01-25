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
	print(type(trainingset))
	print(trainingset[0])
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


















# output = [None]*10 # used to store outputs and evaluate the prediction of network
# accuracy = pandas.DataFrame(0,index=range(0,EPOCH_MAX+1),columns=['test_c','test_i','train_c','train_i'])

# # set up confusion matrix: rows=actual, col=predicted
# confu  = pandas.DataFrame(0,index=range(0,10),columns=range(0,10)) 

# # Import data
# train_data = pandas.read_csv(TRAIN_FILE,header=None)
# test_data  = pandas.read_csv(TEST_FILE ,header=None)

# # Preprocess data 
# train_data.sample(frac=1)      # shuffle training data
# train_target = train_data[0].values # Save targets as a separate dataframe/array
# train_data.drop(columns=0)     # Remove column with target info
# train_data = train_data.values # convert to numpy array
# train_data = numpy.divide(train_data, INPUT_MAX) # scale inputs between 0 and 1 by dividing by input max value

# test_target = test_data[0].values # Save targets as a separate dataframe/array
# test_data.drop(columns=0)    # Remove column with target info
# test_data = test_data.values # convert to numpy array
# test_data = numpy.divide(test_data, INPUT_MAX)

# input_size = len(train_data[0]) # how many inputs are there
