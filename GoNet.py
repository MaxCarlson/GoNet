from __future__ import print_function
import numpy as np
import cntk as cntk
from cntk.ops import relu
from cntk.initializer import he_normal, normal
from cntk.layers import MaxPooling, BatchNormalization, Dense, Dropout, Convolution2D
from DataGenerator import Generator
from Globals import BoardDepth, BoardLength, BoardLengthP, BoardSize, BoardSizeP
import scipy

batchSize = 256
maxEpochs = 30
featurePath = "./data/features"
labelPath = "./data/labels"
saveDir = './SavedModels/'
netName = 'GoNet'

def Conv(input, filterShape, filters,  strides=(1,1), activation=True, padding=True):
    cn = Convolution2D(filterShape, filters, activation=None, pad=padding)(input)
    ba = BatchNormalization()(cn) #map_rank=1, normalization_time_constant=4096
    do = Dropout(0.22)(ba)

    return relu(do) if activation else do

def ResStack(input, filters):
    c0 = Conv(input, (3,3), filters, 1)
    c1 = Conv(c0,    (3,3), filters, 1, activation=False)
    return relu(c1 + input)

def goNet(input, filters, outSize):

    c0 = Conv(input, (3,3), filters, 1)
    r0 = ResStack(c0, filters)
    r1 = ResStack(r0, filters)
    r2 = ResStack(r1, filters)
    
    pl = MaxPooling((2,2), (2,2))(r2)
    r3 = ResStack(pl, filters)
    r4 = ResStack(r3, filters)
    r5 = ResStack(r4, filters)

    c1 = Conv(r5, (1,1), 2, 1)
    z  = Dense(outSize, activation=None)(c1)

    #GraphViz error, FIX!
    #print(cntk.logging.plot(z, filename='./graph.svg'))

    return z

# Prints the nets accuracy over numFromGen
# batches read from the generator class
def printAccuracy(net, string, g, numFromGen):

    counted = 0
    correct = 0
    for i in range(numFromGen):
        X, Y, W = next(g)
        outs = net(X)
        #outs = cntk.softmax(outs).eval()
        pred = np.argmax(Y, 1)
        indx = np.argmax(outs, 1)
        same = pred == indx
        counted += np.shape(Y)[0]
        correct += np.sum(same)

    print(string, (correct/counted)*100.0)

def trainNet():
    
    gen = Generator(featurePath, labelPath, (0, 10), batchSize, loadSize=3)
    valGen = Generator(featurePath, labelPath, (34, 35), batchSize, loadSize=1)

    filters = 64
    inputVar = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    labelVar = cntk.ops.input_variable((BoardSize), np.float32) 

    net = goNet(inputVar, filters, BoardSize)
   
    # Loss and metric
    loss = cntk.cross_entropy_with_softmax(net, labelVar)
    acc  = cntk.classification_error(net, labelVar)
    
    learner = cntk.adam(net.parameters, 0.33, 0.9, minibatch_size=batchSize) 

    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)
    
    trainer = cntk.Trainer(net, (loss, acc), learner, progressPrinter)

    g = gen.generator()
    vg = valGen.generator()
    for epoch in range(maxEpochs):
        
        miniBatches = 0
        while miniBatches < gen.stepsPerEpoch:
            X, Y, W = next(g)
            miniBatches += 1 # TODO: NEED to make sure this doesn't go over minibatchSize so we're not giving innacurate #'s
            trainer.train_minibatch({inputVar : X, labelVar : Y})

       
        trainer.summarize_training_progress()
        printAccuracy(net, 'Validation Acc %', vg, valGen.stepsPerEpoch)

        if epoch >= 6:
            net.save(saveDir + netName + "_{}.dnn".format(epoch))


trainNet()
