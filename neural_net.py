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
	ones = np.ones(shape=x.shape)
	return np.subtract(ones, x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

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

		# activation vectors
		self.hiddenacts = np.zeros(self.j+1)
		self.hiddenacts[0] = 1.0
		self.outputs = np.zeros(self.k)

		# weight matrices
		self.hiddenweights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(self.j+1, self.i))
		self.hiddenweights[0] = np.zeros(self.i)
		self.hiddenweights[0][0] = 1.0
		# self.hiddenweights = np.array([[1, 0, 0],
		# 					[0.5, -0.05, 0.05],
		# 					[-0.5, -0.25, 0.25],
		# 					[-0.2, -0.5, 0.5]])
		self.outputweights = np.random.uniform(low=WEIGHT_MIN, high=WEIGHT_MAX, size=(self.k, self.j+1))
		# self.outputweights = np.array([[0.1, 0.2, -0.5, 0.3],
        #             		[0.15, -0.05, 0.25, 0.4]])
		self.deltaWj = np.zeros((self.j+1, self.i))
		self.deltaWk = np.zeros((self.k, self.j+1))

		self.targets = np.zeros(self.k)

		self.confmat = np.zeros(shape=(self.k, self.k))
		if self.verbose:
			print('the network was initialized with:')
			print('The number of inputs to this net is:', self.i)
			print('The number of hidden units to this net is:', self.j)
			print('The number of outputs to this net is:', self.k)
			print('and a learning rate of ',self.lrate, ' and a momentum of ', self.momentum)
			print('the initial weights are:')
			print('Hidden Weight Matrix')
			print(self.hiddenweights)
			print('The size of the hidden weight matrix is: ', self.hiddenweights.shape)
			print('Output Weight Matrix')
			print(self.outputweights)
			print('The size of the hidden weight matrix is: ', self.outputweights.shape)
			print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
		return

	def PredictVector(self, data):
		if self.verbose:
			print('\n******************************')
		inputs = np.array(data)
		if self.verbose:
			print('the length of inputs is ', data.shape)
		self.hiddenacts = sigmoid(np.inner(inputs, self.hiddenweights))
		self.hiddenacts[0] = 1
		self.outputs = sigmoid(np.inner(self.hiddenacts, self.outputweights))
		if self.verbose:
			print('by using the following input,')
			print(data)
			print('the network predicted the following:')
			print('hidden activations:\n', self.hiddenacts)
			print('output predictions:\n', self.outputs)
			print('******************************\n')
		return

	def PredictMatrix(self, data):
		if self.verbose:
			print('\n******************************')
		inputs = np.array(data)
		if self.verbose:
			print('the length of inputs is ', data.shape)
		self.hiddenacts = sigmoid(np.inner(inputs, self.hiddenweights))
		self.hiddenacts[:, 0] = np.ones(len(data))
		self.outputs = sigmoid(np.inner(self.hiddenacts, self.outputweights))
		if self.verbose:
			print('by using the following input,')
			print(data)
			print('the network predicted the following:')
			print('hidden activations:\n', self.hiddenacts)
			print('output predictions:\n', self.outputs)
			print('******************************\n')
		return

	def Train(self, data, target):
		if self.verbose:
			print('\n+++++++++++++++++++++++++++++++')
			print('Now training on ')
			print(data)
			print('which has a shape of: ', data.shape)
			print('where the target is:', target)
		self.BuildTarget(target)
		if self.verbose:
			print('this makes the target vector the following,')
			print(self.targets)
		self.PredictVector(data)
		self.BackwardPropigate(data)
		if self.verbose:
			print('to be updated to:')
			print('Hidden Weight Matrix')
			print(self.hiddenweights)
			print('The size of the hidden weight matrix is: ', self.hiddenweights.shape)
			print('Output Weight Matrix')
			print(self.outputweights)
			print('The size of the hidden weight matrix is: ', self.outputweights.shape)
			print('+++++++++++++++++++++++++++++++\n')
		return

	def BuildTarget(self, target):
		for i in range(self.k):
			if target == i:
				self.targets[i] = 0.9
			else:
				self.targets[i] = 0.1
		return

	def BackwardPropigate(self, inputs):
		if self.verbose:
			print('\n;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
		# determine output errors terms
		kerror = np.subtract(self.targets, self.outputs)
		if self.verbose:
			print('now generating the k error term vector')
			print('the result of the target vector minus the output vector is:')
			print(kerror)
			print('with a shape of: ', kerror.shape)
			print('1-K:')
			print(oneminus(self.outputs))
		kerror = np.multiply(oneminus(self.outputs), kerror)
		if self.verbose:
			print('the result of the first element-wise multiplication is:')
			print(kerror)
			print('with a shape of: ', kerror.shape)
		kerror = np.multiply(self.outputs, kerror)
		if self.verbose:
			print('the result of the second element-wise multiplication is:')
			print(kerror)
			print('with a shape of: ', kerror.shape)

		# determine hidden activation error terms
		jerror = np.inner(np.transpose(self.outputweights), kerror)
		if self.verbose:
			print('\nnow generating the j error term vector')
			print('the result of the target vector minus the output vector is:')
			print(jerror)
			print('with a shape of: ', jerror.shape)
			print('1-J:')
			print(oneminus(self.hiddenacts))
		jerror = np.multiply(oneminus(self.hiddenacts), jerror)
		if self.verbose:
			print('the result of the first element-wise multiplication is:')
			print(jerror)
			print('with a shape of: ', jerror.shape)
		jerror = np.multiply(self.hiddenacts, jerror)
		if self.verbose:
			print('the result of the second element-wise multiplication is:')
			print(jerror)
			print('with a shape of: ', jerror.shape)

		# determine delta weights for hidden layer to output
		if self.verbose:
			print('\nNow calculating the adjestments to the output weights')
			print('the previous delta ouput weight matrix:')
			print(self.deltaWk)
		priordeltaWk = np.multiply(self.deltaWk, self.momentum)
		if self.verbose:
			print('after scaling by the momentum')
			print(priordeltaWk)
		kerror = np.multiply(self.lrate, kerror)
		if self.verbose:
			print('scaling the k error vector by the learning rate')
			print(kerror)
			print('the outer multiply of k error vector and the hidden activiations is:')
			print(np.outer(kerror, self.hiddenacts))
		self.deltaWk = np.add(priordeltaWk, np.outer(kerror, self.hiddenacts))
		if self.verbose:
			print('the final delta matrix for the output weights:')
			print(self.deltaWk)
			print('with a shape of: ', self.deltaWk.shape)

		#determine delta weights for input to hidden layer
		if self.verbose:
			print('\nNow calculating the adjestments to the hidden unit weights')
			print('the previous delta hidden unit weight matrix:')
			print(self.deltaWj)
		priordeltaWj = np.multiply(self.momentum, self.deltaWj)
		if self.verbose:
			print('after scaling by the momentum')
			print(priordeltaWj)
		jerror = np.multiply(self.lrate, jerror)
		if self.verbose:
			print('scaling the j error vector by the learning rate')
			print(jerror)
			print('the outer multiply of j error vector and the inputs is:')
			print(np.outer(jerror, inputs))
		self.deltaWj = priordeltaWj + np.outer(jerror, inputs)
		if self.verbose:
			print('the final delta matrix for the hidden unit weights:')
			print(self.deltaWj)
			print('with a shape of: ', self.deltaWj.shape)

		#
		self.outputweights = self.outputweights + self.deltaWk
		self.hiddenweights = self.hiddenweights + self.deltaWj
		if self.verbose:
			print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n')
		return

	def ConfusionMatrix(self, targets):
		if self.verbose:
			print('\n-----------------------------')
		# make confmat
		self.confmat = np.zeros((self.k, self.k))
		# for each row in predictions
		for index in range(len(targets)):
			predicted_index = self.outputs[index].argmax(axis=0)
			#predicted_index = 0
			target_index = targets[index]
			if self.verbose:
				print('incrementing target:', target_index, ' and predicted:',
					  self.outputs[index][predicted_index], 'at index ', predicted_index)
			self.confmat[target_index][predicted_index] += 1
		if self.verbose:
			print('the generated confusion matrix is:')
			print(self.confmat)
			print('-----------------------------\n')
		return

	def GetConfusionMatrix(self):
		return self.confmat

	def Accuracy(self, data, targets):
		if self.verbose:
			print('\n///////////////////////////////')
			print('Now calculating the accuracy of the network on the following data:')
			print(data)
			print('compared to this target array')
			print(targets)
		self.PredictMatrix(data)
		self.ConfusionMatrix(targets)
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
