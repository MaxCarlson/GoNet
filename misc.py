import numpy as np
from Globals import BoardLength
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
# TODO: Figure out how to cycle learning rates
# without using huge amount of memory i.e: integrate with cntk.lr_scheduler
#
# Full Cycle length (cycling from min to max, then down to min)
# is stepMult * itsInEpoch * 2
def learningRateCycles(maxEpoch, cycleLen, minRate, maxRate, itsInEpoch):
    lrs = []
    stepMult = cycleLen // 2
    stepSize = stepMult * itsInEpoch
    for ec in range(maxEpoch // stepMult):
        for it in range(stepSize * 2):
            cycle   = math.floor(1 + it / (2 * stepSize))
            x       = math.fabs(it / stepSize - 2 * cycle + 1) 
            lrs.append(minRate + (maxRate - minRate) * max(0, 1-x))
    return lrs 

# Helper for finding the optimal learning rate
# Provides a list of exponentially increasing values between
# minRate-maxRate in order to look at learning rate/loss corelation
#
# TODO: Provide a utility to plot learning rate against the loss
# instead of relying on tensorboard estimates
def findOptLr(maxEpoch, minRate, maxRate, itsInEpoch):
    lr       = []
    lnMin    = log(minRate)
    totalIts = itsInEpoch * maxEpoch
    step     = (log(maxRate) - lnMin) / totalIts
    for i in range(totalIts):
        tmp = lnMin + i * step
        lr.append(exp(tmp))

    return lr

def rebuildBoard(boards):
    stm     = boards[1]
    # Make opponent 2's
    opp     = boards[2] + boards[2]
    board   = stm + opp
    colAr   = np.zeros((19, 19, 3), np.uint8)
    colAr   += np.uint8([86,47,14]) 
    BLACK   = 1
    WHITE   = 2
    
    # Couldn't figure out how to vectorize this with np
    for x in range(BoardLength):
        for y in range(BoardLength):
            if stm[x][y]    == BLACK:
                colAr[x][y] = [0,0,0]
            elif opp[x][y]  == WHITE:
                colAr[x][y] = [255,255,255]

    return colAr

def netHeatmap(net, gen):

    X, Y, W = next(gen)
    outs    = net(X)
    exIn    = rebuildBoard(X[0])
    exOut   = np.reshape(outs[0][0], (BoardLength, BoardLength))

    fig = plt.figure(frameon=False)
    Z1 = np.add.outer(range(19), range(19)) % 2
    im1  = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest')
    im2 = plt.imshow(exOut, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
    plt.show()
    a = 5

