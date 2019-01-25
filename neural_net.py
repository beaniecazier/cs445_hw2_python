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

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def outerror(o, t)
	return o*(1-o)*(t-o)

def hiddenerror()

class neural_net:
	def __init__(self, hyperparameters):
		pass

	def SetWeights(w, h, weights = None):
		if weights != None:
			return
		#initialize array
		pass

	def PrintActivations():
		print(self.hiddenacts)
		print(self.outputs)
		pass

	def Train(data):
		for inputs in data:
			ForwardPropigate(inputs)
			CalcErrorTerms()
			BackwardPropigate()
		pass

	def ForwardPropigate(inputs):
		np.matmul(a=inputs, b=self.hiddenweights, out=self.hiddendot[1:])
		self.hiddenacts = map(sigmoid, self.hiddendot)
		np.matmul(a=self.hiddenacts, b=self.outputweights, out=self.outputdot)
		self.outputs = map(sigmoid, self.outputdot)
		if self.verbose:
			PrintActivations()
		pass

	def CalcErrorTerms();
		pass

	def BackwardPropigate():
		pass

	def Predict():
		pass