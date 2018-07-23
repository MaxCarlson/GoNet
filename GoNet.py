from __future__ import print_function
import math
#import scipy
import numpy as np
import cntk as cntk
from Net import goNet
from DataGenerator import Generator
from NetHeatMap import NetHeatMap
from misc import printAccuracy, learningRateCycles, findOptLr
from Globals import BoardDepth, BoardLength, BoardSize, BoardSizeP

batchSize = 128
maxEpochs = 20
featurePath = "./data/features"
labelPath = "./data/labels"
saveDir = './SavedModels/'
netName = 'GoNet'

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
        print('Sucessful load of model', loadPath, '\n')
    else:
        net = goNet(inputVar, filters, BoardSize, 2)

    # Comment to forgoe generating heat map
    # of network outputs over input board state
    hmap = NetHeatMap(net, g)
    hmap.genHeatmap(15)
   
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
    #lrs         = learningRateCycles(maxEpochs, cycleLen, 0.0001, 0.00025, gen.stepsPerEpoch)
    # TODO: Use this so we don't have to generate scchedule for every iteration
    #cntk.learners.learning_parameter_schedule(lrs, batchSize, gen.stepsPerEpoch*cycleLen)
    #lrs         = findOptLr(1, 0.00001, 0.01, gen.stepsPerEpoch)
    #Current Best 0.0001-0.00025
    learner     = cntk.adam(net.parameters, 0.0001, epoch_size=batchSize, momentum=0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) 

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



#trainNet()
trainNet('SavedModels/GoNetLeaky_1_47_65_2.477.dnn', True)