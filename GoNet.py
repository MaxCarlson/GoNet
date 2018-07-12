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
maxEpochs = 100
featurePath = "./data/features"
labelPath = "./data/labels"
saveDir = './SavedModels/'
netName = 'GoNet'

def Conv(input, filterShape, filters, activation=True, padding=True):
    cn = Convolution2D(filterShape, filters, activation=None, pad=padding)(input)
    ba = BatchNormalization()(cn) 
    do = Dropout(0.13)(ba)
    return relu(do) if activation else do

def ResLayer(input, filters):
    c0 = Conv(input, (3,3), filters)
    c1 = Conv(c0,    (3,3), filters, activation=False)
    return relu(c1 + input)

def ResStack(input, filters, count):
    inp = input
    for l in range(count):
        inp = ResLayer(inp, filters)
    return inp

def ValueHead(input, size, valueOut):
    vc = Convolution2D((1,1), 1, activation=None)(input)
    b0 = BatchNormalization()(vc)
    dr = Dropout(0.14)(b0)
    r0 = relu(dr)
    d0 = Dense(size, activation=None)(r0)
    do = Dropout(0.14)(d0)
    r1 = relu(do)
    d1 = Dense(valueOut, activation=None)(r1)
    return d1 #cntk.layers.tanh(d1)

def goNet(input, filters, policyOut, valueOut):

    c0 = Conv(input, (3,3), filters) 
    rs = ResStack(c0, filters, 10)

    # Policy Head
    pc = Conv(rs, (1,1), 2, 1)
    p  = Dense(policyOut, activation=None)(pc)

    # Value Head
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
def learningRateCycles(maxEpoch, minRate, maxRate, stepSize):
    
    lrs = []
    for ec in range(maxEpoch):
        cycle = math.floor(1 + ec / (2 * stepSize))
        x = math.fabs(ec / stepSize - 2 * cycle + 1) 
        lrs.append(minRate + (maxRate - minRate) * max(0, 1-x))
    return lrs 

def trainNet(loadPath = '', load = False):
    
    gen     = Generator(featurePath, labelPath, (0, 1), batchSize, loadSize=3)
    valGen  = Generator(featurePath, labelPath, (299, 300), batchSize, loadSize=1)

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
    
    # Old learning rate = 0.04
    #
    lrs     = learningRateCycles(maxEpochs, 0.02, 0.03, 4)
    learner = cntk.adam(net.parameters, lrs, epoch_size=gen.samplesEst, momentum=0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) 

    #cntk.logging.TrainingSummaryProgressCallback()
    #cntk.CrossValidationConfig()

    tbWriter        = cntk.logging.TensorBoardProgressWriter(freq=2, log_dir='./TensorBoard/', model=net)
    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   
    trainer         = cntk.Trainer(net, (loss, error), learner, [progressPrinter, tbWriter])
    
    g  = gen.generator()
    vg = valGen.generator()

    plot = PlotModel()
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

        plot.update(losses, policyAccs, valueAccs)   
        net.save(saveDir + netName + '_{}_{}_{}_{:.3f}.dnn'.format(epoch+1, policyAcc, valueAcc, losses[epoch][1]))







#trainNet()
trainNet('SavedModels/GoNet_1_44_62_2.968.dnn', True)