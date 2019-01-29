import pandas as pd

class Hyperparameters:
	def __init__(self, df):
		self.verbose = bool(df["verbose"])
		self.lrate = df["lrate"]
		self.momentum = df["momentum"]
		self.minweight = df["minweight"]
		self.maxweight = df["maxweight"]
		self.numclasses = df["numclasses"]
		self.numhidden = df["numhidden"]
		self.numinputs = df["numinputs"]