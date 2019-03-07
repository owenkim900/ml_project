import numpy as np
import math
from util_prediction import oneOfK
from util_prediction import predictionAnalysis

class NB:
	def __init__(self, **kwargs):
		self.params = kwargs

	def train(self, training_data, numClass, valueScale):
		numSample = training_data.shape[0]
		dataDim = training_data.shape[1] - 1
		self.priorCount = [0] * numClass
		labelSet = training_data[:,0:1]
		dataSet = training_data[:,1:]
		#estimate prior distribution
		for i in range(0, numSample):
			self.priorCount[int(labelSet[i,0])] += 1
		self.prior = [float(i) / sum(self.priorCount) for i in self.priorCount]
		#estimate likelihood
		self.likelihood = []
		for i in range(0, numClass):
			temp = np.zeros((dataDim, valueScale))
			self.likelihood.append(temp)
		for i in range(0, numSample):
			for j in range(0, dataDim):
				self.likelihood[int(labelSet[i,0])][j, int(dataSet[i,j])] += 1
		for i in range(0, numClass):
			temp = np.sum(self.likelihood[i], axis = 1)[np.newaxis].T
			temp = np.matlib.repmat(temp, 1, valueScale)
			self.likelihood[i] = np.divide(self.likelihood[i],temp)
		print "Training is finished."

	def predict(self, data, numClass):
		predictionDistribution = []
		data = data
		dataDim = data.shape[1]
		for i in range(0, numClass):
			temp = 1.0
			for j in range(0, dataDim):
				temp *=	self.likelihood[i][j, int(data[0,j])]
			predictionDistribution.append(temp * self.prior[i])
		maxlikelihood = 0
		index = 0
		for j in range(0, numClass):
			if predictionDistribution[j] > maxlikelihood:
				maxlikelihood = predictionDistribution[j]
				index = j
		return index		

	def test(self, test_data, numClass):
		numSample = test_data.shape[0]
		testingData = test_data[:, 1:]
		testingLabel = test_data[:, 0:1]
		prediction = np.zeros((numSample,1))
		for i in range(0, numSample):
			prediction[i,0] = self.predict(testingData[i,:], numClass)
		predictMat = oneOfK(prediction, numClass)
		labelMat = oneOfK(testingLabel, numClass)
		predictionAnalysis(predictMat, labelMat)

class GNB:
	def __init__(self, **kwargs):
		self.params = kwargs

	def train(self, training_data, numClass):
		numSample = training_data.shape[0]
		dataDim = training_data.shape[1] - 1
		self.priorCount = [0] * numClass
		labelSet = training_data[:,0:1]
		dataSet = training_data[:,1:]
		#estimate prior distribution
		for i in range(0, numSample):
			self.priorCount[int(labelSet[i,0])] += 1
		self.prior = [float(i) / sum(self.priorCount) for i in self.priorCount]
		#estimate Gaussian distribution
		self.likelihood = []
		for i in range(0, numClass):
			temp = np.zeros((dataDim, 2))
			self.likelihood.append(temp)
		self.count = self.likelihood
		#estimate mean
		for i in range(0, numSample):
			for j in range(0, dataDim):
				self.count[int(labelSet[i,0])][j, 0] += dataSet[i,j]
		for i in range(0, numClass):
			for j in range(0, dataDim):
				self.likelihood[i][j,0] = self.count[i][j,0] / self.priorCount[i]
		#estimate variance
		for i in range(0, numSample):
			for j in range(0, dataDim):
				self.count[int(labelSet[i,0])][j, 1] += pow(dataSet[i,j], 2) - pow(self.likelihood[int(labelSet[i,0])][j,0], 2)
		for i in range(0, numClass):
			for j in range(0, dataDim):
				self.likelihood[i][j,1] = self.count[i][j,1] / (self.priorCount[i] - 1)

		print "Training is finished."

	def predict(self, data, numClass):
		predictionDistribution = []
		data = data
		dataDim = data.shape[1]
		for i in range(0, numClass):
			temp = 1.0
			for j in range(0, dataDim):
				temp *=	(float(1) / math.sqrt(2 * math.pi * self.likelihood[i][j,1])) 
				temp *= math.exp(-pow((data[0,j] - self.likelihood[i][j,0]), 2) / (2 * self.likelihood[i][j,1]))
			predictionDistribution.append(temp * self.prior[i])
		maxlikelihood = 0
		index = 0
		for j in range(0, numClass):
			if predictionDistribution[j] > maxlikelihood:
				maxlikelihood = predictionDistribution[j]
				index = j
		return index		

	def test(self, test_data, numClass):
		numSample = test_data.shape[0]
		testingData = test_data[:, 1:]
		testingLabel = test_data[:, 0:1]
		prediction = np.zeros((numSample,1))
		for i in range(0, numSample):
			prediction[i,0] = self.predict(testingData[i,:], numClass)

		#print prediction - testingLabel
		predictMat = oneOfK(prediction, numClass)
		labelMat = oneOfK(testingLabel, numClass)
		predictionAnalysis(predictMat, labelMat)
		