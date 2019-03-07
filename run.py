#Notice:
#Decision tree on continuous data takes a long time because a large number of threshold settings are tested
import numpy as np
import sys
from load_data import load_iris
from load_data import load_congress_data
from load_data import load_monks
from nn import NN
from nb import NB
from nb import GNB
from tree import DT
import random
import math
import numpy.matlib

#number of class of each data set
ncIris = 3
ncCongress = 2
ncMonks = 2
#Dataset reading
trainSetIris = np.matrix(load_iris(0.7)[0])
testSetIris = np.matrix(load_iris(0.7)[1])
trainSetCongress = np.matrix(load_congress_data(0.7)[0])
testSetCongress = np.matrix(load_congress_data(0.7)[1])
trainM1 = np.matrix(load_monks(1)[0])
testM1 = np.matrix(load_monks(1)[1])
trainM2 = np.matrix(load_monks(2)[0])
testM2 = np.matrix(load_monks(2)[1])
trainM3 = np.matrix(load_monks(3)[0])
testM3 = np.matrix(load_monks(3)[1])

#Decision Tree on Congress using IG
print '1. DT, Congress, IG--------------------------------'
model0 = DT()
model0.train(trainSetCongress, ncCongress, 1)
model0.test(testSetCongress, ncCongress, 1)
#Prune
model0.prune(trainSetCongress, ncCongress)
print '2. -----------prune----------'
model0.test(testSetCongress, ncCongress, 1)
print '--------------------------------'
#Decision Tree on Congress using IGR
print '3. DT, Congress, IGR--------------------------------'
model1 = DT()
model1.train(trainSetCongress, ncCongress, 0)
model1.test(testSetCongress, ncCongress, 1)
#Prune
model1.prune(trainSetCongress, ncCongress)
print '4. -----------prune----------'
model1.test(testSetCongress, ncCongress, 1)
print '--------------------------------'
#Decision Tree on Iris using IG, this takes a while since pow(10, 4) possible splits scheme is tested
print '5. DT, Iris, IG--------------------------------'
model2 = DT()
model2.continuousTrain(trainSetIris, ncIris, 10, 1) # 10 is to divide the range of continuous data into 10 splits for threshold
model2.test(testSetIris, ncIris, 1)
#Prune
model2.prune(trainSetIris, ncIris)
print '6. -----------prune----------'
model2.test(testSetIris, ncIris, 1)
print '--------------------------------'
#Decision Tree on Iris using IGR, this takes a while since pow(10, 4) possible splits scheme is tested
print '7. DT, Iris, IGR--------------------------------'
model3 = DT()
model3.continuousTrain(trainSetIris, ncIris, 10, 0) # 10 is to divide the range of continuous data into 10 splits for threshold
model3.test(testSetIris, ncIris, 1)
#Prune
model3.prune(trainSetIris, ncIris)
print '8. -----------prune----------'
model3.test(testSetIris, ncIris, 1)
print '--------------------------------'
#Decision Tree on Monks using IG, this takes a while since possible splits scheme is tested
print '9. DT, Monks1, IG--------------------------------'
model4 = DT()
model4.discreteTrain(trainM1, ncMonks, 1)
model4.test(testM1, ncMonks, 1)
#Prune
model4.prune(trainM1, ncMonks)
print '10. -----------prune----------'
model4.test(testM1, ncMonks, 1)
print '--------------------------------'
print '11. DT, Monks2, IG--------------------------------'
model5 = DT()
model5.discreteTrain(trainM2, ncMonks, 1)
model5.test(testM2, ncMonks, 1)
#Prune
model5.prune(trainM2, ncMonks)
print '12. -----------prune----------'
model5.test(testM2, ncMonks, 1)
print '--------------------------------'

print '13. DT, Monks3, IG--------------------------------'
model6 = DT()
model6.discreteTrain(trainM3, ncMonks, 1)
model6.test(testM3, ncMonks, 1)
#Prune
model6.prune(trainM3, ncMonks)
print '14. -----------prune----------'
model6.test(testM3, ncMonks, 1)
print '--------------------------------'
#Decision Tree on Monks using IGR, this takes a while since possible splits scheme is tested

print '15. DT, Monks1, IGR--------------------------------'
model7 = DT()
model7.discreteTrain(trainM1, ncMonks, 0)
model7.test(testM1, ncMonks, 1)
print testM1.shape
#Prune
model7.prune(trainM1, ncMonks)
print '16. -----------prune----------'
model7.test(testM1, ncMonks, 1)
print '--------------------------------'
print '17. DT, Monks2, IGR--------------------------------'
model8 = DT()
model8.discreteTrain(trainM2, ncMonks, 0)
model8.test(testM2, ncMonks, 1)
#Prune
model8.prune(trainM2, ncMonks)
print '18. -----------prune----------'
model8.test(testM2, ncMonks, 1)
print '--------------------------------'
print '19. DT, Monks3, IGR--------------------------------'
model9 = DT()
model9.discreteTrain(trainM3, ncMonks, 0)
model9.test(testM3, ncMonks, 1)
#Prune
model9.prune(trainM3, ncMonks)
print '20. -----------prune----------'
model9.test(testM3, ncMonks, 1)
print '--------------------------------'
#Neural Network
print '21. NN, Iris, alternate weight, momentum--------------------------------'
model21 = NN(arch=[5,6, ncIris])
model21.initializeWeight(trainSetIris)
model21.train(trainSetIris, ncIris)
model21.test(testSetIris, ncIris)
print '--------------------------------'
print '22. NN, Iris, not alternate weight, momentum--------------------------------'
model22 = NN(arch=[5,6, ncIris])
model22.train(trainSetIris, ncIris)
model22.test(testSetIris, ncIris)
print '--------------------------------'
print '23. NN, Iris, not alternate weight, no momentum--------------------------------'
model23 = NN(arch=[5,6, ncIris])
model23.turnOffMomentum()
model23.train(trainSetIris, ncIris)
model23.test(testSetIris, ncIris)
print '24. NN, Iris, alternate weight, no momentum--------------------------------'
model24 = NN(arch=[5,6, ncIris])
model24.initializeWeight(trainSetIris)
model24.turnOffMomentum()
model24.train(trainSetIris, ncIris)
model24.test(testSetIris, ncIris)
print '--------------------------------'
print '25. NN, Congress, alternate weight, momentum--------------------------------'
model25 = NN(arch=[5,6, ncCongress])
model25.initializeWeight(trainSetCongress)
model25.train(trainSetCongress, ncCongress)
model25.test(testSetCongress, ncCongress)
print '--------------------------------'
print '26. NN, Congress, no alternate weight, no momentum--------------------------------'
model26 = NN(arch=[5,6, ncCongress])
model26.turnOffMomentum()
model26.train(trainSetCongress, ncCongress)
model26.test(testSetCongress, ncCongress)
print '--------------------------------'
print '27. NN, Congress, no alternate weight, momentum--------------------------------'
model27 = NN(arch=[5,6, ncCongress])
model27.train(trainSetCongress, ncCongress)
model27.test(testSetCongress, ncCongress)
print '--------------------------------'
print '28. NN, Congress, alternate weight, no momentum--------------------------------'
model28 = NN(arch=[5,6, ncCongress])
model28.initializeWeight(trainSetCongress)
model28.turnOffMomentum()
model28.train(trainSetCongress, ncCongress)
model28.test(testSetCongress, ncCongress)
print '--------------------------------'
print '29. NN, Monks1, alternate weight, momentum--------------------------------'
model29 = NN(arch=[5,6, ncMonks])
model29.initializeWeight(trainM1)
model29.train(trainM1, ncMonks)
model29.test(testM1, ncMonks)
print '--------------------------------'
print '30. NN, Monks1, not alternate weight, momentum--------------------------------'
model30 = NN(arch=[5,6, ncMonks])
model30.train(trainM1, ncMonks)
model30.test(testM1, ncMonks)
print '--------------------------------'
print '31. NN, Monks1, not alternate weight, no momentum--------------------------------'
model31 = NN(arch=[5,6, ncMonks])
model31.turnOffMomentum()
model31.train(trainM1, ncMonks)
model31.test(testM1, ncMonks)
print '32. NN, Monks1, alternate weight, no momentum--------------------------------'
model32 = NN(arch=[5,6, ncMonks])
model32.initializeWeight(trainM1)
model32.turnOffMomentum()
model32.train(trainM1, ncMonks)
model32.test(testM1, ncMonks)
print '--------------------------------'
print '33. NN, Monks2, alternate weight, momentum--------------------------------'
model33 = NN(arch=[5,6, ncMonks])
model33.initializeWeight(trainM2)
model33.train(trainM2, ncMonks)
model33.test(testM2, ncMonks)
print '--------------------------------'
print '34. NN, Monks2, not alternate weight, momentum--------------------------------'
model34 = NN(arch=[5,6, ncMonks])
model34.train(trainM2, ncMonks)
model34.test(testM2, ncMonks)
print '--------------------------------'
print '35. NN, Monks2, not alternate weight, no momentum--------------------------------'
model35 = NN(arch=[5,6, ncMonks])
model35.turnOffMomentum()
model35.train(trainM2, ncMonks)
model35.test(testM2, ncMonks)
print '36. NN, Monks2, alternate weight, no momentum--------------------------------'
model36 = NN(arch=[5,6, ncMonks])
model36.initializeWeight(trainM2)
model36.turnOffMomentum()
model36.train(trainM2, ncMonks)
model36.test(testM2, ncMonks)
print '--------------------------------'
print '37. NN, Monks3, alternate weight, momentum--------------------------------'
model37 = NN(arch=[5,6, ncMonks])
model37.initializeWeight(trainM3)
model37.train(trainM3, ncMonks)
model37.test(testM3, ncMonks)
print '--------------------------------'
print '38. NN, Monks3, not alternate weight, momentum--------------------------------'
model38 = NN(arch=[5,6, ncMonks])
model38.train(trainM3, ncMonks)
model38.test(testM3, ncMonks)
print '--------------------------------'
print '39. NN, Monks3, not alternate weight, no momentum--------------------------------'
model39 = NN(arch=[5,6, ncMonks])
model39.turnOffMomentum()
model39.train(trainM3, ncMonks)
model39.test(testM3, ncMonks)
print '40. NN, Monks3, alternate weight, no momentum--------------------------------'
model40 = NN(arch=[5,6, ncMonks])
model40.initializeWeight(trainM3)
model40.turnOffMomentum()
model40.train(trainM3, ncMonks)
model40.test(testM3, ncMonks)
print '--------------------------------'
print '41. NB, Congress--------------------------------'
model41 = NB()
model41.train(trainSetCongress, ncCongress, 3)
model41.test(testSetCongress, ncCongress)
print '--------------------------------'
print '42. GNB, Congress--------------------------------'
model42 = GNB()
model42.train(trainSetCongress, ncCongress)
model42.test(testSetCongress, ncCongress)
print '--------------------------------'
print '43. GNB, Iris--------------------------------'
model43 = GNB()
model43.train(trainSetIris, ncIris)
model43.test(testSetIris, ncIris)
print '--------------------------------'
print '44. NB, Monks1--------------------------------'
model44 = NB()
model44.train(trainM1, ncMonks, 5)
model44.test(testM1, ncMonks)
print '--------------------------------'
print '45. GNB, Monks1--------------------------------'
model45 = GNB()
model45.train(trainM1, ncMonks)
model45.test(testM1, ncMonks)
print '--------------------------------'
print '46. NB, Monks2--------------------------------'
model46 = NB()
model46.train(trainM2, ncMonks, 5)
model46.test(testM2, ncMonks)
print '--------------------------------'
print '47. GNB, Monks2--------------------------------'
model47 = GNB()
model47.train(trainM2, ncMonks)
model47.test(testM2, ncMonks)
print '--------------------------------'
print '48. NB, Monks3--------------------------------'
model48 = NB()
model48.train(trainM3, ncMonks, 5)
model48.test(testM3, ncMonks)
print '--------------------------------'
print '49. GNB, Monks3--------------------------------'
model49 = GNB()
model49.train(trainM3, ncMonks)
model49.test(testM3, ncMonks)
print '--------------------------------'
