import numpy as np
import math
from math       import exp, log
from Globals    import BoardLength
import matplotlib.pyplot as plt


# Prints the nets accuracy over numFromGen
# batches read from the generator class
#
# TODO: Figure out how to get this done with
# cntk.classification_error so it's on the GPU!
def printAccuracy(net, string, g, numFromGen):

    counted = 0
    pcorrect = 0
    vcorrect = 0
    for i in range(numFromGen):
        X, Y, W = next(g)
        outs = net(X)
        
        # PolcyNetworkAcc
        pred = np.argmax(Y, 1)
        indx = np.argmax(outs[0], 1)
        psame = pred == indx

        # Value Network Acc
        pred = np.argmax(W, 1)
        indx = np.argmax(outs[1], 1)
        vsame = pred == indx

        counted += np.shape(Y)[0]
        pcorrect += np.sum(psame)
        vcorrect += np.sum(vsame)

    valueAcc  = (vcorrect/counted)*100.0
    policyAcc = (pcorrect/counted)*100.0
    print(string)
    print('PolicyAcc', policyAcc)
    print('ValueAcc',  valueAcc)

    return int(policyAcc), int(valueAcc)

# Cycle the learning rate linearly between min and max rates
# for apparently faster/better training. 
# https://arxiv.org/pdf/1506.01186.pdf
#
# Doesn't handle permanant decreases in learning rate 
#
# Full Cycle length (cycling from min to max, then down to min)
# is stepMult * itsInEpoch * 2
def learningRateCycles(cycleLen, minRate, maxRate, itsInEpoch, startMax = True):
    lrs = []
    cycleLen = int(cycleLen)
    stepMult = cycleLen // 2
    stepSize = stepMult * itsInEpoch
    for it in range(stepSize * 2):
        cycle   = math.floor(1 + it / (2 * stepSize))
        x       = math.fabs(it / stepSize - 2 * cycle + 1) 
        lrs.append(minRate + (maxRate - minRate) * max(0, 1-x))

    # Start at the high end of the cycle. 
    # This is useful if we crashed/stopped after the first half
    # of a cycle. 
    # TODO: This should be saved and loaded info with the model instead of manually determined
    if startMax:
        lrs = lrs[stepSize:stepSize*2] + lrs[0:stepSize]

    return lrs 

# Helper for finding the optimal learning rate
# Provides a list of exponentially increasing values between
# minRate-maxRate in order to look at learning rate/loss corelation
#
# TODO: Provide a utility to plot learning rate against the loss
# instead of relying on tensorboard estimates
def findOptLr(maxEpoch, minRate, maxRate, itsInEpoch):
    lr       = []
    minRate  = float(minRate)
    maxRate  = float(maxRate)
    lnMin    = log(minRate)
    totalIts = itsInEpoch * maxEpoch
    step     = (log(maxRate) - lnMin) / totalIts
    for i in range(totalIts):
        tmp = lnMin + i * step
        lr.append(exp(tmp))

    return lr