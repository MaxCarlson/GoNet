from __future__ import print_function
import numpy as np
import cntk as cntk
from cntk.ops import relu
from DataGenerator import Generator
from cntk.layers import MaxPooling, BatchNormalization, Dense, Dropout, Convolution2D
from Globals import BoardDepth, BoardLength, BoardLengthP, BoardSize, BoardSizeP
#import scipy

batchSize = 128
maxEpochs = 30
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

def trainNet(loadPath = '', load = False):
    
    gen = Generator(featurePath, labelPath, (0, 69), batchSize, loadSize=3)
    valGen = Generator(featurePath, labelPath, (70, 71), batchSize, loadSize=1)

    filters = 64
    inputVar = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    policyVar = cntk.ops.input_variable((BoardSize), np.float32)
    valueVar = cntk.ops.input_variable((2), np.float32) 

    net = cntk.placeholder()
    
    if load == True:
        net = cntk.load_model(loadPath)
    else:
        net = goNet(inputVar, filters, BoardSize, 2)
   
    # Loss and metric
    policyLoss = cntk.cross_entropy_with_softmax(net.outputs[0], policyVar)
    valueLoss = cntk.cross_entropy_with_softmax(net.outputs[1], valueVar)
    loss = policyLoss + valueLoss

    policyError = cntk.element_not(cntk.classification_error(net.outputs[0], policyVar))
    valueError = cntk.element_not(cntk.classification_error(net.outputs[1], valueVar))
    #error = (valueError + policyError) / 2
    error = valueError
    
    # Initial learning rate = 0.05
    #
    learner = cntk.adam(net.parameters, 0.04, 0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) 

    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)
    
    trainer = cntk.Trainer(net, (loss, error), learner, progressPrinter)

    g = gen.generator()
    vg = valGen.generator()
    for epoch in range(maxEpochs):
        
        miniBatches = 0
        while miniBatches < gen.stepsPerEpoch:
            X, Y, W = next(g)
            miniBatches += 1 # TODO: NEED to make sure this doesn't go over minibatchSize so we're not inputting more than we're saying we are
            trainer.train_minibatch({net.arguments[0] : X, policyVar : Y, valueVar : W}) 

       
        trainer.summarize_training_progress()
        policyAcc, valueAcc = printAccuracy(net, 'Validation Acc %', vg, valGen.stepsPerEpoch)

        net.save(saveDir + netName + '_{}_{}_{}.dnn'.format(epoch+1, policyAcc, valueAcc))


trainNet()
#trainNet('SavedModels/GoNet_39_55.dnn', True)