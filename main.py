# Design Unit/Module: Multiclass Perceptron
# File Name: main.py
# Description: Use the MNIST data set to train and test a network of 10 perceptrons.
# Assumptions: First entry in each row contains target class
# Limitations: None?
# System: any(Python)
# Author: Preston Cazier
# Course: CS 445 Machine Learning (Winter 2019)
# Assignment: Homework 1
# Revision: 1.0 03/1/2019

from neural_net import neural_net
import numpy as np
import pandas as pd

# initialize hyperparameter variables
EPOCHS = 50
INPUT_MAX = 255
LRATE = 0.1
MOMENTUM = [.9, 0.0, 0.5, 1.0]
NUMCLASSES = 10
NUMHIDDEN = [10,50,100]
VERBOSE = False
TRAIN_FILE = "mnist_train.csv"
TEST_FILE = "mnist_test.csv"

# set up Accuracy recording
Accuracy = pd.DataFrame(0.0, index=range(
	0, EPOCHS+1), columns=['test', 'train'])

# set up confusion matrix: rows=actual, col=predicted
confmat_train = pd.DataFrame(0, index=range(0, 10), columns=range(0, 10))
confmat_test = pd.DataFrame(0, index=range(0, 10), columns=range(0, 10))

# load data
train_data = pd.read_csv(TRAIN_FILE, header=None)
test_data = pd.read_csv(TEST_FILE, header=None).values
# # Preprocess data
train_data = train_data.sample(frac=1).values
if VERBOSE:
	print('now randomizing the training data and separating out the targets')

# Save targets as a separate dataframe/array
train_target = train_data[:, 0]
train_data = np.array(train_data)
train_data[:, 0] = INPUT_MAX
if VERBOSE:
	print('training data targets:')
	print(train_target)
	print('training data:')
	print(train_data)
	print('the shape of this set of data is:')
	print(train_data.shape)
	print('now separating out the targets from the testing data')
test_target = test_data[:, 0]
test_data = np.array(test_data)
test_data[:, 0] = INPUT_MAX
if VERBOSE:
	print('testing data targets:')
	print(test_target)
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

# for i in range(len(NUMHIDDEN)):
# 	print('Starting Experiment 1 with ', NUMHIDDEN[i], ' hidden units')
# 	# initialize neural network
# 	network = neural_net(numclasses=NUMCLASSES, numhidden=NUMHIDDEN[i], numinputs=input_size,
# 						momentum=MOMENTUM[0], lrate=LRATE, verbose=VERBOSE)
	
# 	for j in range(1, EPOCHS+1):
# 		print('starting epoch ', j)
# 		for k in range(len(train_data)):
# 			network.Train(train_data[k], train_target[k])
# 		print('finding Accuracy')
# 		Accuracy['test'][j] = network.Accuracy(test_data, test_target)
# 		Accuracy['train'][j] = network.Accuracy(train_data, train_target)

# 	# Generate the final confusion matrices and print the data
# 	confmat_test = network.GetConfusionMatrix()
# 	confmat_train = network.GetConfusionMatrix()
# 	Accuracy['test'].to_csv('acc_test_'+str(NUMHIDDEN[i])+'_'+str(LRATE)+'.csv')
# 	Accuracy['train'].to_csv('acc_train_'+str(NUMHIDDEN[i])+'_'+str(LRATE)+'.csv')
# 	pd.DataFrame(confmat_train).to_csv('confmat_train_'+str(NUMHIDDEN[i])+'_'+str(LRATE)+'.csv')
# 	pd.DataFrame(confmat_test).to_csv('confmat_test_'+str(NUMHIDDEN[i])+'_'+str(LRATE)+'.csv')


for i in range(1, len(MOMENTUM)):
	print('Starting Experiment 2 with a momentum of ', MOMENTUM[i])
	# initialize neural network
	network = neural_net(numclasses=NUMCLASSES, numhidden=NUMHIDDEN[2], numinputs=input_size,
					  momentum=MOMENTUM[i], lrate=LRATE, verbose=VERBOSE)

	for j in range(1, EPOCHS+1):
		print('starting epoch ', j)
		for k in range(len(train_data)):
			network.Train(train_data[k], train_target[k])
		print('finding Accuracy')
		Accuracy['test'][j] = network.Accuracy(test_data, test_target)
		Accuracy['train'][j] = network.Accuracy(train_data, train_target)

	# Generate the final confusion matrices and print the data
	confmat_test = network.GetConfusionMatrix()
	confmat_train = network.GetConfusionMatrix()
	Accuracy['test'].to_csv('acc_test_m'+str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
	Accuracy['train'].to_csv('acc_train_m'+str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
	pd.DataFrame(confmat_train).to_csv('confmat_train_m' +
									str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
	pd.DataFrame(confmat_test).to_csv('confmat_test_m' +
									str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')


print('Starting Experiment 3 with a quarter of the data')
# create a quarter slice of data
split = int(len(train_data) / 4)
quarter_train_data = train_data[:split, :]
quarter_train_target = train_target[0:split]
# initialize neural network
network = neural_net(numclasses=NUMCLASSES, numhidden=NUMHIDDEN[2], numinputs=input_size,
					momentum=MOMENTUM[0], lrate=LRATE, verbose=VERBOSE)

for j in range(1, EPOCHS+1):
	print('starting epoch ', j)
	for k in range(len(quarter_train_data)):
		network.Train(quarter_train_data[k], quarter_train_target[k])
	print('finding Accuracy')
	Accuracy['test'][j] = network.Accuracy(test_data, test_target)
	Accuracy['train'][j] = network.Accuracy(
		quarter_train_data, quarter_train_target)

# Generate the final confusion matrices and print the data
confmat_test = network.GetConfusionMatrix()
confmat_train = network.GetConfusionMatrix()
Accuracy['test'].to_csv('quarter_acc_test_m'+str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
Accuracy['train'].to_csv('quarter_acc_train_m'+str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
pd.DataFrame(confmat_train).to_csv('quarter_confmat_train_m' +
								str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
pd.DataFrame(confmat_test).to_csv('quarter_confmat_test_m' +
								  str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')


print('Starting Experiment 3 with half of the data')
#create a half slice of data
split = int(len(train_data) / 2)
half_train_data = train_data[:split, :]
half_train_target = train_target[0:split]
# initialize neural network
network = neural_net(numclasses=NUMCLASSES, numhidden=NUMHIDDEN[2], numinputs=input_size,
					 momentum=MOMENTUM[i], lrate=LRATE, verbose=VERBOSE)

for j in range(1, EPOCHS+1):
	print('starting epoch ', j)
	for k in range(len(half_train_data)):
		network.Train(half_train_data[k], half_train_target[k])
	print('finding Accuracy')
	Accuracy['test'][j] = network.Accuracy(test_data, test_target)
	Accuracy['train'][j] = network.Accuracy(half_train_data, half_train_target)

# Generate the final confusion matrices and print the data
confmat_test = network.GetConfusionMatrix()
confmat_train = network.GetConfusionMatrix()
Accuracy['test'].to_csv('half_acc_test_m'+str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
Accuracy['train'].to_csv('half_acc_train_m'+str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
pd.DataFrame(confmat_train).to_csv('half_confmat_train_m' +
								   str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
pd.DataFrame(confmat_test).to_csv('half_confmat_test_m' +
								  str(MOMENTUM[i])+'_'+str(LRATE)+'.csv')
