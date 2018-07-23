import numpy as np
import math
from math import exp, log
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

def addStoneToImg(boardImg, x, y, stoneImg):
    shp = np.shape(stoneImg)
    offX = x * shp[0]
    offY = y * shp[1]
    boardImg[offX:offX+shp[0], offY:offY+shp[1]] = stoneImg

def buildBoardImg(board, boardImg, w, b):
    BLACK   = 1
    WHITE   = 2
    for x in range(BoardLength):
        for y in range(BoardLength):
            img = None
            if board[x, y] == BLACK:
                img = b
            elif board[x, y] == WHITE:
                img = w
            else:
                continue
            addStoneToImg(boardImg, x, y, img)

    return boardImg

# Convert the output image to the same
# shape as the board image with pieces
def buildOutImage(netOut, shp):
    outImg = np.zeros((BoardLength*shp[0],BoardLength*shp[1]))

    for x in range(BoardLength):
        for y in range(BoardLength):
            offX = x * shp[0]
            offY = y * shp[1]
            outImg[offX:offX+shp[0], offY:offY*shp[1]] = netOut[x,y]
    return outImg

def buildImages(boards, netOut):

    # Build the nets input image (with pieces and such)
    stm     = boards[1]
    opp     = boards[2] + boards[2]
    board   = stm + opp
    w       = plt.imread('./img/shellStone.png')
    b       = plt.imread('./img/blackStone.png')
    shp     = np.shape(w)
    boardImg    = np.zeros((BoardLength*shp[0],BoardLength*shp[1],shp[2]))
    boardImg    = buildBoardImg(board, boardImg, w, b)
    outImg      = buildOutImage(netOut, shp)

    return boardImg, outImg

def netHeatmap(net, gen):

    X, Y, W = next(gen)
    outs    = net(X)
    exOut   = np.reshape(outs[0][0], (BoardLength, BoardLength))

    imgIn, imgOut = buildImages(X[0], exOut)

    fig     = plt.figure(frameon=False)
    plt.imshow(imgIn)
    plt.imshow(imgOut, cmap=plt.cm.viridis, alpha=.5, interpolation='bilinear')

    #Z1      = np.add.outer(range(19), range(19)) % 2
    #im1     = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest')
    #im2     = plt.imshow(exIn)
    #im3     = plt.imshow(exOut, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
    plt.show()
    a = 5

