from __future__ import print_function
import numpy as np
import cntk as cntk
from cntk.ops import relu
from cntk.initializer import he_normal, normal
from cntk.layers import MaxPooling, BatchNormalization, Dense, Dropout, Convolution2D
from DataGenerator import Generator
from Globals import BoardDepth, BoardLength, BoardLengthP, BoardSize, BoardSizeP
import scipy


#import cntk.tests.test_utils
#cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for CNTK internal build system)

batchSize = 512
maxEpochs = 100
featurePath = "./data/features"
labelPath = "./data/labels"

def Conv(input, filterShape, filters,  strides=(1,1), padding=False):
    cn = Convolution2D(filterShape, filters, activation=None, pad=padding)(input)
    ba = BatchNormalization()(cn) #map_rank=1, normalization_time_constant=4096
    do = Dropout(0.2)(ba)
    return relu(do)

def DenseL(input, outSize):
    de = Dense(outSize, activation=None)(input)
    ba = BatchNormalization()(de)
    do = Dropout(0.15)(ba)
    return relu(do)

def goNet(input, filters, outSize):
    
    # TODO: Look into what this padding turns out to actually pad to
    # See if we can get padding similar to AlphaGo's, i.e. pad board to 23x23
    c0 = Conv(input, (5, 5), filters, padding=True)
    c1 = Conv(c0, (3, 3), filters)
    c2 = Conv(c1, (3, 3), filters, padding=True)
    c3 = Conv(c2, (3, 3), filters, padding=True)
    c4 = Conv(c3, (3, 3), filters, padding=True)
    pool0 = MaxPooling((2,2))(c4)

    c5 = Conv(pool0, (3, 3), filters, padding=True)
    c6 = Conv(c5,    (3, 3), filters, padding=True)
    pool1 = MaxPooling((2,2))(c6)

    y = DenseL(pool1, outSize)
    z = DenseL(y,    outSize)

    #GraphViz error, FIX!
    #print(cntk.logging.plot(z, filename='./graph.svg'))

    return z

# Prints the nets accuracy on whatever inputs are given
# 
def printAccuracy(net, X, Y):
    outs = net(X)
    pred = np.argmax(Y, 1)
    indx = np.argmax(outs, 1)
    same = pred == indx
    print("Accuracy %", np.sum(same)/batchSize*100)

def trainNet():
    
    gen = Generator(featurePath, labelPath, (0, 15), batchSize)

    inputVar = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    labelVar = cntk.ops.input_variable(BoardSize, np.float32) 

    net = goNet(inputVar, 64, BoardSize)
   
    # Loss and metric
    loss = cntk.cross_entropy_with_softmax(net, labelVar)
    acc  = cntk.classification_error(net, labelVar)
    
    minisPerBatch = gen.stepsPerEpoch()
    learner = cntk.adam(net.parameters, 0.0018, 0.9, minibatch_size=None) # minibatch_size=batchSize ?
    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)
    
    trainer = cntk.Trainer(net, (loss, acc), learner, progressPrinter)

    g = gen.generator()
    X, Y = np.ones(1), np.ones(1)
    for epoch in range(maxEpochs):
        
        miniBatches = 0
        while miniBatches < minisPerBatch:
            X, Y = next(g)
            miniBatches += 1 # TODO: NEED to make sure this doesn't go over minibatchSize so we're not giving innacurate #'s

            trainer.train_minibatch({inputVar : X, labelVar : Y})

       
        trainer.summarize_training_progress()
        printAccuracy(net, X, Y)

        


trainNet()

