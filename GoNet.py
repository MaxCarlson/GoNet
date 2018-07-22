from __future__ import print_function
import math
#import scipy
import numpy as np
import cntk as cntk
from Net import goNet
from math import exp, log
from DataGenerator import Generator
from win10toast import ToastNotifier
from Globals import BoardDepth, BoardLength, BoardSize, BoardSizeP


# TODO: Not working!
def displayNotifs(epoch, cycleLen):
    title   = 'Epoch {} complete'.format(epoch)
    msgStr  = ''
    if (epoch + 1) % cycleLen == 0:
        msgStr = 'Cycle {} complete!'.format(epoch // cycleLen)

    notifier = ToastNotifier()
    notifier.show_toast(title, duration=100, threaded=True)


batchSize = 128
maxEpochs = 20
featurePath = "./data/features"
labelPath = "./data/labels"
saveDir = './SavedModels/'
netName = 'GoNet'

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

def findOptLrExp(maxEpoch, minRate, maxRate, itsInEpoch):
    lr       = []
    lnMin    = log(minRate)
    totalIts = itsInEpoch * maxEpoch
    step     = (log(maxRate) - lnMin) / totalIts
    for i in range(totalIts):
        tmp = lnMin + i * step
        lr.append(exp(tmp))

    return lr

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
    
    cycleLen    = 2
    #lrs         = learningRateCycles(maxEpochs, cycleLen, 0.006, 0.0068, gen.stepsPerEpoch)
    # TODO: Use this so we don't have to generate scchedule for every iteration
    #cntk.learners.learning_parameter_schedule(lrs, batchSize, gen.stepsPerEpoch*cycleLen)
    #lrs         = findOptLr(1, 0.001, 0.07, gen.stepsPerEpoch//3)
    learner     = cntk.adam(net.parameters, 0.006, epoch_size=batchSize, momentum=0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) 

    #cntk.logging.TrainingSummaryProgressCallback()
    #cntk.CrossValidationConfig()

    # TODO: Figure out how to write multiple 'metrics'
    tbWriter        = cntk.logging.TensorBoardProgressWriter(freq=1, log_dir='./TensorBoard/', model=net)
    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   
    trainer         = cntk.Trainer(net, (loss, error), learner, [progressPrinter, tbWriter])
    
    ls          = []
    losses      = []
    #valueAccs   = []
    #policyAccs  = []

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
        #policyAccs.append([epoch, policyAcc])
        #valueAccs.append([epoch, valueAcc])   


        # TODO: When loading a model, make sure to save it with epoch+previousModelEpoch
        # so that we can have contiguous epoch counters on save&load
        net.save(saveDir + netName + 'Leaky_{}_{}_{}_{:.3f}.dnn'.format(epoch+1, policyAcc, valueAcc, losses[epoch][1]))
        displayNotifs(epoch, cycleLen)



#trainNet()
trainNet('SavedModels/GoNetLeaky_1_47_64_2.557.dnn', True)