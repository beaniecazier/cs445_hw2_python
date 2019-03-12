# 1d array of input m values
# 1d array of hidden layer n+1 activations(the values that are input to output layer)
# 1d array of putput layer k activations
# 1d array of k length for targets (ex.target class is 9 => arr is .1 .1 .1 .1 .1 .1 .1 .1 .9 .1 for a 10 class NN)
# Nx(M+1) array of input to hidden layer weights
# Kx(N+1) array of hidden layer to output layer weights
# 1d array of k length output errors
# 1d array of n+1 length hidden errors

import pandas as pd
import numpy as np
import math

WEIGHT_MIN = -0.05
WEIGHT_MAX = 0.05

def oneminus(x): 
	return 1-x

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class neural_net:
	def __init__(self, numclasses, numhidden, numinputs, momentum, lrate, verbose):
		# hyperparameters
		self.verbose = verbose
		if self.verbose:
			print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
			print('now initializing the network')
		self.lrate = lrate
		self.momentum = momentum
		self.minweight = WEIGHT_MIN
		self.maxweight = WEIGHT_MAX
		self.k = numclasses
		self.j = numhidden
		self.i = numinputs
		if self.verbose:
			print('The number of inputs to this net is:', self.i)
			print('The number of hidden units to this net is:', self.j)
			print('The number of outputs to this net is:', self.k)

		# activation vectors
		self.hiddenacts = np.zeros(self.j+1)
		self.hiddenacts[0] = 1.0
		self.outputs = np.zeros(self.k)

		# weight matrices
		self.hiddenweights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(self.j+1, self.i+1))
		self.hiddenweights[0] = np.zeros(self.i+1)
		self.hiddenweights[0][0] = 1.0
		print(self.hiddenweights)
		print(self.hiddenweights.shape)
		self.outputweights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(self.k, self.j+1))
		self.deltaWj = np.zeros((self.j+1, self.i+1))
		self.deltaWk = np.zeros((self.k, self.j+1))

		self.targets = np.zeros(self.k)

		self.confmat = np.zeros(shape=(self.k, self.k))
		return

	def BackwardPropigate(self, inputs):
		# determine output errors terms
		kerror = np.multiply(map(oneminus, self.outputs), np.multiply(self.targets - self.outputs))
		kerror = np.multiply(self.outputs, kerror)

		# determine hidden activation error terms
		jerror = np.matmul(a=kerror,b=self.outputweights)
		jerror = np.multiply(map(oneminus, self.hiddenacts), jerror)
		jerror = np.multiply(self.hiddenacts, kerror)

		# determine delta weights for hidden layer to output
		kerror = np.multiply(kerror, self.lrate)
		priordeltaWk = np.multiply(self.deltaWk, self.momentum)
		self.detlaWk = priordeltaWk + np.outer(a=kerror, b=self.hiddenacts)

		#determine delta weights for input to hidden layer
		jerror = np.multiply(jerror, self.lrate)
		priordeltaWj = np.multiply(self.deltaWj, self.momentum)
		self.detlaWj = priordeltaWj + np.matmul(a=jerror, b=inputs)

		#
		self.outputweights = self.outputweights + self.detlaWk
		self.hiddenweights = self.hiddenweights + self.detlaWj
		return

	def BuildTarget(self, target):
		for i in range(10):
			if target == i:
				self.targets[i] = 0.9
			else:
				self.targets[i] = 0.1
		return

	def Train(self, data, target):
		if self.verbose:
			print('\n+++++++++++++++++++++++++++++++')
			print('Now training on ')
			print(data)
		self.BuildTarget(target)
		self.Predict(data)
		self.BackwardPropigate(data)
		if self.verbose:
			print('to be updated to:')
			print(self.hiddenweights)
			print(self.outputweights)
			print('+++++++++++++++++++++++++++++++\n')
		return

	def Predict(self, data):
		if self.verbose:
			print('\n******************************')
		inputs = np.array(data)
		if self.verbose:
			print('the length of inputs is ', data.shape)
		self.hiddenacts = map(sigmoid, np.matmul(a=self.hiddenweights, b=inputs))
		self.outputs = map(sigmoid, np.matmul(
			a=self.outputweights, b=self.hiddenacts))
		if self.verbose:
			print('by using the following input,')
			print(data)
			print(self.hiddenacts)
			print('the network predicted the following:')
			print(self.outputs)
			print('******************************\n')
		return

	def ConfusionMatrix(self, predictions, targets):
		return self.confmat

	def Accuracy(self, data, targets):
		if self.verbose:
			print('\n///////////////////////////////')
			print('Now calculating the accuracy of the network on the following data:')
			print(data)
			print('compared to this target array')
			print(targets)
		self.Predict(data)
		self.ConfusionMatrix(self.outputs, targets)
		# sum diagonal to get accuracy
		total = 0
		for i in range(self.k):
			total += self.confmat[i][i]
		acc = total / len(data)
		if self.verbose:
			print('this gives tp count of ', total, 'out of ',
				  len(data), 'for an accuracy of ', acc)
			print('///////////////////////////////\n')
		return acc
