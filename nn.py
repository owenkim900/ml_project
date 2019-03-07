import numpy as np
import sys
from load_data import load_iris
from load_data import load_congress_data
from load_data import load_monks
from util_prediction import oneOfK
from util_prediction import predictionAnalysis
from util_prediction import getPrediction
import util_prediction
import random
import math
import numpy.matlib

class NN:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.momentum = 1

    def initializeWeight(self, training_data):
        numpy.random.seed(1)
        self.architecture = []
        for i in self.params['arch']:
            if not len(self.architecture) == 0:
                uniformRange = float(2)/ math.sqrt(self.architecture[-1][0].shape[1])
                self.architecture.append([np.random.uniform(0, uniformRange, (self.architecture[-1][0].shape[1],i)) - 0.5 * uniformRange, np.zeros((1,i))])
            else:
                uniformRange = float(2)/ math.sqrt(training_data.shape[1] - 1)
                self.architecture.append([np.random.uniform(0, uniformRange, (training_data.shape[1] - 1,i)) - 0.5 * uniformRange, np.zeros((1,i))])
    
    def turnOffMomentum(self):
        self.momentum = 0

    def train(self, training_data, numClass):
        numpy.random.seed(2)
        numSample = training_data.shape[0]
        if 'architecture' not in self.__dict__:
            self.architecture = []
            for i in self.params['arch']:
                uniformRange = 0.2
                if not len(self.architecture) == 0:
                    self.architecture.append([np.random.uniform(0, uniformRange, (self.architecture[-1][0].shape[1],i)) - 0.5 * uniformRange, np.zeros((1,i))])
                else:
                    self.architecture.append([np.random.uniform(0, uniformRange, (training_data.shape[1] - 1,i)) - 0.5 * uniformRange, np.zeros((1,i))])
        for p in range(0, 500):
            self.flow = [training_data[:,1:]]
            self.delta = []
        #Forwardpropagation
            for i in range(0, len(self.architecture)):
                et = np.exp(np.dot(self.flow[i], self.architecture[i][0]) + np.matlib.repmat(self.architecture[i][1], numSample, 1))
                self.flow.append(et / ( 1 + et))
            delta = self.flow[len(self.architecture)] - oneOfK(training_data[:,0:1], numClass)
            df = np.multiply(self.flow[len(self.architecture)], 1 - self.flow[len(self.architecture)])
            temp = np.multiply(delta, df)
            self.delta = []
            for i in range(0, len(self.architecture)):
                self.delta.append(0)
            self.delta[len(self.architecture) - 1] = temp
        #Compute delta of each layer
            currentLayer = len(self.architecture) - 2
            while currentLayer >= 0:
                self.delta[currentLayer] = np.dot(self.delta[currentLayer + 1], self.architecture[currentLayer + 1][0].T)
                self.delta[currentLayer] = np.multiply(self.delta[currentLayer], np.multiply(self.flow[currentLayer + 1], 1 - self.flow[currentLayer + 1]))
                currentLayer -= 1
        #Compute Gradient of each sample
            self.dW = []
            self.db = []
            for j in range(0, len(self.architecture)):
                tempDW = []
                for k in range(0, numSample):
                    tempDW.append(np.dot(self.flow[j][k:(k+1),:].T, self.delta[j][k:(k+1),:]))
                self.dW.append(tempDW)
                self.db.append(self.delta[j])
        #Update weights and biases
            r = 0.05
            rho = 0
            if self.momentum == 1:
                rho = 0.1
            for k in range(0, len(self.architecture)):
                tempW = 0
                tempB = 0
                for kk in range(0, numSample):
                    tempW = rho * tempW - r * self.dW[k][kk]
                    tempB = rho * tempB - r * self.db[k][kk:(kk+1),:]
                    self.architecture[k][0] += tempW
                    self.architecture[k][1] += tempB

    def predict(self, data, numClass):
        self.flow = [data[:,1:]]
        for i in range(0, len(self.architecture)):
            self.flow.append(1 / ( 1 + np.exp(-np.dot(self.flow[-1], self.architecture[i][0]) - np.matlib.repmat(self.architecture[i][1], numSample, 1))))
        return getPrediction(self.flow[-1])

    def test(self, test_data, numClass):
        numSample = test_data.shape[0]
        self.flow = [test_data[:,1:]]
        for i in range(0, len(self.architecture)):
            self.flow.append(1 / ( 1 + np.exp(-np.dot(self.flow[i], self.architecture[i][0]) - np.matlib.repmat(self.architecture[i][1], numSample, 1))))
        predictMat = getPrediction(self.flow[-1])
        labelMat = oneOfK(test_data[:,0],numClass)
        #Compute presicion
        predictionAnalysis(predictMat, labelMat)
