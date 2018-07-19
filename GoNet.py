from __future__ import print_function
import math
#import scipy
import numpy as np
import cntk as cntk
from cntk.ops import relu
from DataGenerator import Generator
from cntk.layers import MaxPooling, BatchNormalization, Dense, Dropout, Convolution2D
from Globals import BoardDepth, BoardLength, BoardLengthP, BoardSize, BoardSizeP

batchSize = 128
maxEpochs = 20
featurePath = "./data/features"
labelPath = "./data/labels"
saveDir = './SavedModels/'
netName = 'GoNet'

def Conv(input, filterShape, filters, activation=True, padding=True):
    cn = Convolution2D(filterShape, filters, activation=None, pad=padding)(input)
    ba = BatchNormalization()(cn) 
    #do = Dropout(0.13)(ba)
    # TODO: Try LeakRelu
    #return relu(do) if activation else do
    return cntk.leaky_relu(ba) if activation else ba

# Two convolutional layers with a skip
# connection from input to the seconds layers output
def ResLayer(input, filters):
    c0 = Conv(input, (3,3), filters)
    c1 = Conv(c0,    (3,3), filters, activation=False)
    #return relu(c1 + input)
    return cntk.leaky_relu(c1 + input)

def ResStack(input, filters, count):
    inp = input
    for l in range(count):
        inp = ResLayer(inp, filters)
    return inp

# One Head of the network for predicting whether
# an input will result in a win for side to move or not
def ValueHead(input, size, vOutSize):
    vc = Convolution2D((1,1), 1, activation=None)(input)
    b0 = BatchNormalization()(vc)
    dr = Dropout(0.14)(b0)
    #r0 = relu(dr)
    r0 = cntk.leaky_relu(dr)
    d0 = Dense(size, activation=None)(r0)
    do = Dropout(0.14)(d0)
    #r1 = relu(do)
    r1 = cntk.leaky_relu(do)
    d1 = Dense(vOutSize, activation=None)(r1)
    return d1 

# TODO: Possibly apply a softmax to this as well as ValueHead
# That or just apply it inside Gopher
def PolicyHead(input, pOutSize):
    #pc = Conv(rs, (1,1), 2, 1)
    #p  = Dense(policyOut, activation=None)(pc)
    pc = Conv(input, (1,1), 2, 1)
    return Dense(pOutSize, activation=None)(pc)

def goNet(input, filters, policyOut, valueOut):

    c0 = Conv(input, (3,3), filters) 
    rs = ResStack(c0, filters, 10)

    p = PolicyHead(rs, policyOut)
    v = ValueHead(rs, 128, valueOut)
    
    return cntk.combine(p, v)

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
# without using huge amount of memory
#
# Full Cycle length (cycling from min to max, then down to min)
# is stepMult * itsInEpoch * 2
def learningRateCycles(maxEpoch, minRate, maxRate, itsInEpoch, stepMult = 2):
    lrs = []
    stepSize = stepMult * itsInEpoch
    for ec in range(maxEpoch // stepMult):
        for it in range(stepSize * 2):
            cycle   = math.floor(1 + it / (2 * stepSize))
            x       = math.fabs(it / stepSize - 2 * cycle + 1) 
            lrs.append(minRate + (maxRate - minRate) * max(0, 1-x))
    return lrs 

def trainNet(loadPath = '', load = False):
    
    # Instantiate generators for both training and
    # validation datasets. Grab their generator functions
    tFileShp = (0, 743)
    vFileShp = (744, 745)
    gen      = Generator(featurePath, labelPath, tFileShp, batchSize, loadSize=3)
    valGen   = Generator(featurePath, labelPath, vFileShp, batchSize, loadSize=1)
    g        = gen.generator()
    vg       = valGen.generator()

    filters     = 64
    inputVar    = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    policyVar   = cntk.ops.input_variable((BoardSize), np.float32)
    valueVar    = cntk.ops.input_variable((2), np.float32) 

    net = cntk.placeholder() 

    if load == True:
        net = cntk.load_model(loadPath)
        print('Sucessful load of model ', loadPath, '\n')
    else:
        net = goNet(inputVar, filters, BoardSize, 2)
   
    # Loss and accuracy
    policyLoss  = cntk.cross_entropy_with_softmax(net.outputs[0], policyVar)
    valueLoss   = cntk.cross_entropy_with_softmax(net.outputs[1], valueVar)
    loss        = policyLoss + valueLoss

    # TODO: Figure out how to display/report both errors
    policyError = cntk.element_not(cntk.classification_error(net.outputs[0], policyVar))
    valueError  = cntk.element_not(cntk.classification_error(net.outputs[1], valueVar))
    #error      = (valueError + policyError) / 2
    error       = valueError
    
    lrs     = learningRateCycles(maxEpochs, 0.06, 0.09, gen.stepsPerEpoch)
    learner = cntk.adam(net.parameters, lrs, epoch_size=batchSize, momentum=0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) # Old net 0.007lr

    #cntk.logging.TrainingSummaryProgressCallback()
    #cntk.CrossValidationConfig()

    # TODO: Figure out how to write multiple 'metrics'
    tbWriter        = cntk.logging.TensorBoardProgressWriter(freq=10, log_dir='./TensorBoard/', model=net)
    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   
    trainer         = cntk.Trainer(net, (loss, error), learner, [progressPrinter, tbWriter])
    
    ls          = []
    losses      = []
    valueAccs   = []
    policyAccs  = []

    for epoch in range(maxEpochs):
        
        miniBatches = 0
        while miniBatches < gen.stepsPerEpoch:
            X, Y, W = next(g)
            miniBatches += 1 
            trainer.train_minibatch({net.arguments[0] : X, policyVar : Y, valueVar : W}) 
            ls.append(trainer.previous_minibatch_loss_average)


        trainer.summarize_training_progress()
        policyAcc, valueAcc = printAccuracy(net, 'Validation Acc %', vg, valGen.stepsPerEpoch)
        
        losses.append([epoch, sum(ls) / gen.stepsPerEpoch])
        ls.clear()
        policyAccs.append([epoch, policyAcc])
        valueAccs.append([epoch, valueAcc])   
        
        net.save(saveDir + netName + 'Leaky_{}_{}_{}_{:.3f}.dnn'.format(epoch+1, policyAcc, valueAcc, losses[epoch][1]))



#trainNet()
trainNet('SavedModels/GoNetLeaky_3_40_58_3.025.dnn', True)