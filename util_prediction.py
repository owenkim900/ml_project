import numpy as np



def oneOfK(labelValue, classDim):
    numSample = labelValue.shape[0]
    output = np.zeros((numSample, classDim))
    for x in range(0, numSample):
        l = int(labelValue[x, 0])
        output[x, l] = 1
    return output

def getPrediction(distribution):
    output = np.zeros((distribution.shape[0], distribution.shape[1]))
    for i in range(0, distribution.shape[0]):
        index = 0
        maxdist = 0
        for j in range(0, distribution.shape[1]):
            if distribution[i,j] > maxdist:
                maxdist = distribution[i,j]
                index = j
        output[i,index] = 1
    return output

def predictionAnalysis(predictMat, labelMat):
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    numSample = predictMat.shape[0]
    numClass = predictMat.shape[1]
    total = np.zeros((1, numClass))
    positive = np.zeros((1, numClass))
    for i in range(0, numClass):
        for j in range(0, numSample):
            if predictMat[j,i] == 1.0:
                total[0,i] += 1
                if labelMat[j,i] == 1.0:
                    positive[0,i] += 1
    print "Precision"
    print np.divide(positive, total)
    print positive
    print total

    total = np.zeros((1,1))
    positive = np.zeros((1,1))
    for j in range(0, numSample):
        for i in range(0, numClass):
            if predictMat[j,i] == 1.0:
                total[0,0] += 1
                if labelMat[j,i] == 1.0:
                    positive[0,0] += 1
    print "Accurary"
    print np.divide(positive, total)
    print positive
    print total

    total = np.zeros((1, numClass))
    positive = np.zeros((1, numClass))
    for i in range(0, numClass):
        for j in range(0, numSample):
            if labelMat[j,i] == 1.0:
                total[0,i] += 1
                if predictMat[j,i] == 1.0:
                    positive[0,i] += 1
    print "Recall"
    print np.divide(positive, total)
    print positive
    print total