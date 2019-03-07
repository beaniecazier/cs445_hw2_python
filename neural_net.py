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
		self.lrate = lrate
		self.momentum = momentum
		self.minweight = WEIGHT_MIN
		self.maxweight = WEIGHT_MAX
		self.k = numclasses
		self.j = numhidden
		self.i = numinputs

		# activation vectors
		self.hiddenacts = np.zeros(self.j+1)
		self.hiddenacts[0] = 1.0
		self.outputs = np.zeros(self.k)

		# weight matrices
		self.hiddenweights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(self.j+1, self.i+1))
		self.outputweights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(self.k, self.j+1))
		self.deltaWj = np.zeros((self.j+1, self.i+1))
		self.deltaWk = np.zeros((self.k, self.j+1))

		self.targets = np.zeros(self.k)
		return

	def ForwardPropigate(self, inputs):
		self.hiddenacts = map(sigmoid, np.matmul(a=self.hiddenweights, b=inputs))
		self.outputs = map(sigmoid, np.matmul(a=self.outputweights, b=self.hiddenacts))
		if self.verbose:
			print(self.hiddenacts)
			print(self.outputs)
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
		self.BuildTarget(target)
		for inputs in data:
			self.ForwardPropigate(inputs)
			self.BackwardPropigate(inputs)
		return

	def Predict(self, target):
		self.BuildTarget(target)
		return

	def ConfusionMatrix(self, predictions, targets):
		return

	def Accuracy(self, data, targets):
		return
