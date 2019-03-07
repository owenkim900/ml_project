import numpy as np
import math
import copy
from scipy import stats
from util_prediction import oneOfK
from util_prediction import predictionAnalysis

def crossEntropy(set):
	output = 0
	if not set is None:
		numSample = set.shape[0]
		pattern = {}
		for i in range(0,numSample):
			if set[i,0] in pattern:
				pattern[set[i,0]] += 1
			else:
				pattern[set[i,0]] = 1
		for i in pattern:
			output -= (float(pattern[i]) / numSample) * math.log(float(pattern[i]) / numSample)
	return output

class Node:
	def __init__(self, a, layer, index = None, parent=None):
		self.parent = parent
		self.left = None
		self.right = None
		self.testIndex = index 
		self.associate = a #according to the self.testIndex, what group of data is divided to this node
		self.layer = layer

	def getLayer(self):
		return self.self.layer

	def getAssociate(self):
		return self.associate

	def getParent(self):
		return self.parent

	def getLeft(self):
		return self.left

	def getRight(self):
		return self.right

	def getFeature(self):
		return self.testIndex
	
	def setLeft(self, node):
		self.left = node

	def setRight(self, node):
		self.right = node

class DT:
	def __init__(self, **kwargs):
		self.params = kwargs
		self.threshold = None

	def setThreshold(self, s):
		self.threshold = s

	def train(self, training_data, numClass, useGain):
		dataDim = training_data.shape[1] - 1
		self.testFeature = []
		root = Node(training_data, 0, 0)
		self.layer = [[root]]
		candidate = range(1, dataDim + 1)
		divide = 1
		self.leaf = []
		if self.threshold == None:
			self.threshold = [0] * (dataDim + 1)
		while divide:
			#Compute cross entropy of each node
			crossE = []
			for node in self.layer[-1]:
				crossE.append(crossEntropy(node.getAssociate()))
			if sum(crossE) == 0:
				break
			#compute information gain of each candidate feature
			infoGain = []
			SplitInfo = []
			for i in range(0, dataDim + 1):
				infoGain.append(0)
				SplitInfo.append(0)
			for i in candidate:
				for j in range(0, len(self.layer[-1])):
					#compute information gain of node j in the latest self.layer
					if not crossE[j] == 0:
						division0 = None
						division1 = None
						for k in range(0, self.layer[-1][j].getAssociate().shape[0]):
							testValue = float(self.layer[-1][j].getAssociate()[k, i])
							if testValue > self.threshold[i]:
								if division1 is None:
									division1 = self.layer[-1][j].getAssociate()[k,:]
								else:
									division1 = np.vstack((division1, self.layer[-1][j].getAssociate()[k,:]))
							else:
								if division0 is None:
									division0 = self.layer[-1][j].getAssociate()[k,:]
								else:
									division0 = np.vstack((division0, self.layer[-1][j].getAssociate()[k,:]))
					dg = 0
					if division0 is None:
						dg = crossEntropy(division1)
					elif division1 is None:
						dg = crossEntropy(division0)
					else:
						p0 = (float(division0.shape[0]) / (division0.shape[0] + division1.shape[0]))
						p1 = (float(division1.shape[0]) / (division0.shape[0] + division1.shape[0]))
						dg = (float(division0.shape[0]) / (division0.shape[0] + division1.shape[0])) * crossEntropy(division0) 
						+ (float(division1.shape[0]) / (division0.shape[0] + division1.shape[0])) * crossEntropy(division1)
						SplitInfo[i] += - p0 * math.log(p0) - p1 * math.log(p1)
					infoGain[i] += crossE[j] - dg
					if SplitInfo[i] == 0:
						SplitInfo[i] = 0.000001
			#choose feature of the maximum information gain
			index = 1
			maxIG = 0
			for i in candidate:
				if useGain == 1:
					if infoGain[i] > maxIG:
						maxIG = infoGain[i]
						index = i
				else:
					if float(infoGain[i]) / SplitInfo[i] > maxIG:
						maxIG = float(infoGain[i]) / SplitInfo[i]
						index = i
			if sum(infoGain) == 0 or maxIG == 0:
				break
			self.testFeature.append(index)
			candidate.remove(float(index))
			if len(candidate) == 0:
				break
			else:
				collector = []
				for j in range(0, len(self.layer[-1])):
					division0 = None
					division1 = None
					for k in range(0, self.layer[-1][j].getAssociate().shape[0]):
						testValue = float(self.layer[-1][j].getAssociate()[k, index])
						if testValue > self.threshold[index]:
							if division1 is None:
								division1 = self.layer[-1][j].getAssociate()[k,:]
							else:
								division1 = np.vstack((division1, self.layer[-1][j].getAssociate()[k,:]))
						else:
							if division0 is None:
								division0 = self.layer[-1][j].getAssociate()[k,:]
							else:
								division0 = np.vstack((division0, self.layer[-1][j].getAssociate()[k,:]))
					if not division0 == None and not crossE[j] == 0:
						d0 = Node(division0, len(self.layer), index, self.layer[-1][j])
						self.layer[-1][j].setLeft(d0)
						collector.append(d0)
					if not division1 == None and not crossE[j] == 0:
						d1 = Node(division1, len(self.layer), index, self.layer[-1][j])
						self.layer[-1][j].setRight(d1)
						collector.append(d1)
				self.layer.append(collector)
		
		#according to the position in the last self.layer, compute its direction in each previous branch
		self.ruleList = []
		for i in range(1, len(self.layer)):
			for j in range(0, len(self.layer[i])):
				if self.layer[i][j].getLeft() == None and self.layer[i][j].getRight() == None:
					path = []
					group = self.layer[i][j].getAssociate()
					groupNode = self.layer[i][j]
					#Get the label of this group
					label = group[:,0:1]
					count = [0] * numClass
					for k in range(0,label.shape[0]):
						count[int(label[k,0])] += 1
					index = 0
					maxCount = 0
					for k in range(0, numClass):
						if count[k] > maxCount:
							maxCount = count[k]
							index = k
					prediction = index
					#Get the path to this group
					p = groupNode.getParent()
					t = 0
					if p.getRight() == groupNode:
						t = 1
					path.insert(0, [groupNode.getFeature(), t])
					groupNode = p
					while not groupNode.getParent()	== None:
						p = groupNode.getParent()
						t = 0
						if p.getRight() == groupNode:
							t = 1
						path.insert(0, [groupNode.getFeature(), t])
						groupNode = p
					#Add [path, label] to the prediction list
					self.ruleList.append([path, prediction])
		#self.ruleList is a list of judgement conditions, of which each is in the format [[[featureIndex, value],[featureIndex, value],...], label]

	def predict(self, data):
		index = 0
		for condition in self.ruleList:
			found = 0
			for node in condition[0]:
				if node[1] == 1:
					if data[0, int(node[0] - 1)] <= self.threshold[int(node[0])]:
						break
				else:
					if data[0, int(node[0] - 1)] > self.threshold[int(node[0])]:
						break
				if node == condition[0][-1]:
					index = condition[1]
					found = 1
			if found == 1:
				break
		return index		

	def test(self, test_data, numClass, showAnalysis):
		numSample = test_data.shape[0]
		testingData = test_data[:, 1:]
		testingLabel = test_data[:, 0:1]
		prediction = np.zeros((numSample,1))
		for i in range(0, numSample):
			prediction[i,0] = self.predict(testingData[i,:])		
		predictMat = oneOfK(prediction, numClass)
		labelMat = oneOfK(testingLabel, numClass)
		self.error = np.sum(np.abs(predictMat - labelMat)) / 2
		if showAnalysis == 1:
			predictionAnalysis(predictMat, labelMat)

	def continuousTrain(self, training_data, numClass, numDivision, useGain):
		#Get range of data in each feature
		training_data = np.matrix(training_data)
		self.featureMax = np.amax(training_data, axis=0)
		self.featureMin = np.amin(training_data, axis=0)
		self.featureRange = self.featureMax - self.featureMin
		numFeature = training_data.shape[1] - 1
		#Set a set of thresholds
		emptyThreshold = np.zeros((numDivision, 1))
		candidateList = [[0]]
		for i in range(0, numFeature):
			originList = copy.deepcopy(candidateList)
			for j in range(0, numDivision):
				for k in range(0, len(originList)):
					temp = copy.deepcopy(originList[k])
					temp.append(self.featureRange[0,i] * float(j + 1)/ (numDivision + 1))
					candidateList.append(temp)
			for i in originList:
				candidateList.remove(i)
		#Train on each of the threshold setting to fix the best feature and threshold selection
		trainError = []
		for i in candidateList:
			self.setThreshold(i)
			self.train(training_data, numClass, useGain)
			self.test(training_data, numClass, 0)
			trainError.append(self.error)
		index = 0
		minError = training_data.shape[0]
		for i in range(0, len(trainError)):
			if trainError[i] < minError:
				index = i
				minError = trainError[i]
		self.threshold = candidateList[index]
		self.train(training_data, numClass, useGain)

	def discreteTrain(self, training_data, numClass, useGain):
		#Get setting of data in each feature
		training_data = np.matrix(training_data)
		numFeature = training_data.shape[1] - 1
		valueSetting = [[]]
		for i in range(0, numFeature):
			valueSetting.append([])
		for i in range(0, training_data.shape[0]):
			for j in range(1, training_data.shape[1]):
				temp = training_data[i, j]
				if valueSetting[j].count(temp) == 0:
					valueSetting[j].append(temp)
		for i in valueSetting:
			i.sort()

		#Set a set of thresholds
		candidateList = [[0]]
		for i in range(1, numFeature + 1):
			originList = copy.deepcopy(candidateList)
			for j in range(0, len(valueSetting[i])):
				for k in range(0, len(originList)):
					temp = copy.deepcopy(originList[k])
					temp.append(valueSetting[i][j] - 0.0001)
					candidateList.append(temp)
			for i in originList:
				candidateList.remove(i)
		#Train on each of the threshold setting to fix the best feature and threshold selection
		trainError = []
		for i in candidateList:
			self.setThreshold(i)
			self.train(training_data, numClass, useGain)
			self.test(training_data, numClass, 0)
			trainError.append(self.error)
		index = 0
		minError = training_data.shape[0]
		for i in range(0, len(trainError)):
			if trainError[i] < minError:
				index = i
				minError = trainError[i]
		self.threshold = candidateList[index]
		self.train(training_data, numClass, useGain)

	def reconsturct(self, numClass, pruneSet):
		newRule = []
		deleteRule = []
		for i in range(0, len(self.ruleList)):
			if self.ruleList[i][0][-1][0] in pruneSet:
				index = 0
				for j in self.layer[-1]:
					if j.getFeature() == self.ruleList[i][0][-1][0]:
						parentIndex = j.getParent().getFeature();
						temp = j.getParent().getAssociate()
						for k in self.layer[-2]:
							if k.getFeature() == parentIndex and not k == j.getParent():
								temp = np.vstack([temp, k.getAssociate()])
						prediction = []
						for k in range(0, numClass):
							prediction.append(0)
						for k in range(0, temp.shape[0]):
							prediction[int(temp[k,0])] += 1
						maxCount = 0
						for k in range(0, len(prediction)):
							if prediction[k] > maxCount:
								maxCount = prediction[k]
								index = k
						addRule = copy.deepcopy(self.ruleList[i])
						addRule[0].remove(addRule[0][-1])
						addRule[1] = index
						#self.layer[-1].remove(j)
						deletR = copy.deepcopy(self.ruleList[i])
						deleteRule.append(deletR);
						if addRule not in newRule:
							newRule.append(addRule)
		for i in deleteRule:
			while self.ruleList.count(i) > 0:
				self.ruleList.remove(i)
		for i in newRule:
			self.ruleList.append(i)

	def prune(self, training_data, numClass):
		#Compute number of samples of each class, Nc
		originLength = len(self.ruleList)
		numSample = training_data.shape[0]
		dof = len(self.testFeature) - 1
		Nc = []
		for i in range(0, numClass):
			Nc.append(0)
		for i in range(0, training_data.shape[0]):
			Nc[int(training_data[i, 0])] += 1
		#Test Chi-square distribution
		pruneSet = []
		for i in self.testFeature:
			#Count number of samples in terms of feature and class range
			Nx = 0
			Nxc = []
			for j in range(0, numClass):
				Nxc.append(0)
			for j in range(0, training_data.shape[0]):
				if training_data[j, i] > self.threshold[i]:
					Nx += 1
					Nxc[int(training_data[j, 0])] += 1
			#Compute estimation of Nxc
			NxcE = []
			for j in range(0, numClass):
				NxcE.append(float(Nx * Nc[j]) / numSample) 
				NxcE.append(float((numSample - Nx) * Nc[j]) / numSample)
			#Compute deviation
			dev = 0
			for j in range(0, numClass):
				dev += float(math.pow(Nxc[j] - NxcE[2 * j],2)) / NxcE[2 * j]
				dev += float(math.pow(Nc[j] - Nxc[j] - NxcE[2 * j + 1],2)) / NxcE[2 * j + 1]
			#test and prune
			pValue = 1 - stats.chi2.cdf(dev, dof)
			if dev < 3:
				self.testFeature.remove(i)
				pruneSet.append(i)
				'''
				print Nc
				print Nxc
				print Nx
				print numSample
				print self.threshold
				'''
		if len(pruneSet) > 0:
			self.reconsturct(numClass, pruneSet)
		newLength = len(self.ruleList)
		print 'Origin length of rule list is:', originLength, '. After pruning, length of rule list is:', newLength











