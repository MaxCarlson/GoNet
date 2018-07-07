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

def Conv(input, filterShape, filters, activation=True, padding=True):
    cn = Convolution2D(filterShape, filters, activation=None, pad=padding)(input)
    ba = BatchNormalization()(cn) #map_rank=1, normalization_time_constant=4096
    do = Dropout(0.32)(ba)

    return relu(do) if activation else do

def ResStack(input, filters):
    c0 = Conv(input, (3,3), filters)
    c1 = Conv(c0,    (3,3), filters, activation=False)
    return relu(c1 + input)


def ValueHead(input, size, valueOut):
    vc = Convolution2D((1,1), 1, activation=None)(input)
    b0 = BatchNormalization()(vc)
    dp = Dropout(0.45)(b0)
    r0 = relu(dp)

    #re = ResStack(input, 64)
    
    d0 = Dense(size, activation=None)(r0)
    dr = Dropout(0.4)(d0)
    r1 = relu(dr)
    d1 = Dense(valueOut, activation=None)(r1)
    return d1 #cntk.layers.tanh(d1)

def goNet(input, filters, policyOut, valueOut):

    c0 = Conv(input, (3,3), filters)
    r0 = ResStack(c0, filters)
    r1 = ResStack(r0, filters)
    r2 = ResStack(r1, filters)
    
    #pl = MaxPooling((2,2), (2,2))(r2)
    #r3 = ResStack(pl, filters)
    #r4 = ResStack(r3, filters)
    #r5 = ResStack(r4, filters)

    # Policy Head
    pc = Conv(r2, (1,1), 2, 1)
    p  = Dense(policyOut, activation=None)(pc)

    # Value Head
    v = ValueHead(r2, 128, valueOut)
    
    #GraphViz error, FIX!
    #print(cntk.logging.plot(z, filename='./graph.svg'))

    return cntk.combine(p, v)

# Prints the nets accuracy over numFromGen
# batches read from the generator class
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

    print(string)
    print('PolicyAcc', (pcorrect/counted)*100.0)
    print('ValueAcc',  (vcorrect/counted)*100.0)

def trainNet():
    
    gen = Generator(featurePath, labelPath, (0, 15), batchSize, loadSize=3)
    valGen = Generator(featurePath, labelPath, (34, 35), batchSize, loadSize=1)

    filters = 64
    inputVar = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    policyVar = cntk.ops.input_variable((BoardSize), np.float32)
    valueVar = cntk.ops.input_variable((2), np.float32) 
    

    net = goNet(inputVar, filters, BoardSize, 2)
   
    # Loss and metric
    policyLoss = cntk.cross_entropy_with_softmax(net.outputs[0], policyVar)
    valueLoss = cntk.cross_entropy_with_softmax(net.outputs[1], valueVar)
    loss = (policyLoss + valueLoss + valueLoss) / 2

    policyError = cntk.element_not(cntk.classification_error(net.outputs[0], policyVar))
    valueError = cntk.element_not(cntk.classification_error(net.outputs[1], valueVar))
    error = valueError #+ policyError) / 2
    
    learner = cntk.adam(net.parameters, 0.33, 0.99, minibatch_size=batchSize, l1_regularization_weight=0.0002) 

    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)
    
    trainer = cntk.Trainer(net, (loss, error), learner, progressPrinter)

    g = gen.generator()
    vg = valGen.generator()
    for epoch in range(maxEpochs):
        
        miniBatches = 0
        while miniBatches < gen.stepsPerEpoch:
            X, Y, W = next(g)
            miniBatches += 1 # TODO: NEED to make sure this doesn't go over minibatchSize so we're not giving innacurate #'s
            trainer.train_minibatch({inputVar : X, policyVar : Y, valueVar : W})

       
        trainer.summarize_training_progress()
        printAccuracy(net, 'Validation Acc %', vg, valGen.stepsPerEpoch)

        if epoch >= 6:
            net.save(saveDir + netName + "_{}.dnn".format(epoch))


trainNet()
