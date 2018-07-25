from __future__ import print_function
import os
import math
import glob
import numpy as np
import cntk as cntk
from Net import goNet
from DataGenerator import Generator
from NetHeatMap import NetHeatMap
from argparse import ArgumentParser
from misc import printAccuracy, learningRateCycles, findOptLr
from Globals import BoardDepth, BoardLength, BoardSize, BoardSizeP

batchSize = 128
maxEpochs = 50
defaultLr = 0.01
# TODO: Command line args
featurePath = "./data/features"
labelPath = "./data/labels"
saveDir = './SavedModels/'
netName = 'GoNet'

def findLatestModel(loadName):
    latestModel = loadName
    if loadName == 'latest':
        models = glob.glob(saveDir + '*')
        latestModel = max(models, key=os.path.getctime)
    
    return latestModel

def loadModel(args):
    net       = cntk.placeholder() 
    modelName = findLatestModel(args.load)

    if modelName != None:
        net = cntk.load_model(modelName)
        print('Sucessful load of model', modelName, '\n')
    else:
        net = goNet(inputVar, filters, BoardSize, 2)
        print('Created new network!')

    return net

def trainNet(args): #loadPath = '', load = False
    
    # Instantiate generators for both training and
    # validation datasets. Grab their generator functions
    # TODO: Command line args
    tFileShp = (0,1)#(0, 743)
    vFileShp = (5,6)#(744, 745)
    gen      = Generator(featurePath, labelPath, tFileShp, batchSize, loadSize=3)
    valGen   = Generator(featurePath, labelPath, vFileShp, batchSize, loadSize=1)
    g        = gen.generator()
    vg       = valGen.generator()

    filters     = 64
    inputVar    = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    policyVar   = cntk.ops.input_variable((BoardSize), np.float32)
    valueVar    = cntk.ops.input_variable((2), np.float32) 

    net = loadModel(args)

    # Show a heatmap of network outputs 
    # over an input board state
    if args.heatMap:
        hmap = NetHeatMap(net, g)
        hmap.genHeatmap(args.heatMap)
   
    # Loss and accuracy
    policyLoss  = cntk.cross_entropy_with_softmax(net.outputs[0], policyVar)
    valueLoss   = cntk.cross_entropy_with_softmax(net.outputs[1], valueVar)
    loss        = policyLoss + valueLoss

    # TODO: Figure out how to display/report both errors
    policyError = cntk.element_not(cntk.classification_error(net.outputs[0], policyVar))
    valueError  = cntk.element_not(cntk.classification_error(net.outputs[1], valueVar))
    #error      = (valueError + policyError) / 2
    error       = valueError

    lrc = args.lr
    if args.cycleLr[0]:
        lrc = learningRateCycles(1, args.cycleLr[0], args.cycleLr[1], args.cycleLr[2], gen.stepsPerEpoch*args.cycleLr[0])
    elif args.optLr:
        lrc = findOptLr(maxEpochs, args.optLr[0], args.optLr[1], gen.stepsPerEpoch)


    lrc = cntk.learners.learning_parameter_schedule(lrc, batchSize, gen.stepsPerEpoch)

    learner = cntk.adam(net.parameters, lrc, epoch_size=batchSize, momentum=0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) 

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

def parseArgs():
    parser = ArgumentParser()

    # TOP TODO: Auto load latest saved model
    global maxEpochs

    parser.add_argument('-epochs', help='Max # of epochs to train for', type=int, default=maxEpochs)
    parser.add_argument('-lr', help='Set learning rate', type=float, default=defaultLr)
    parser.add_argument('-cycleLr', help='Cycle learning rate between inp1-inp2, input 0 is cycle length', nargs=3, default=[2,.01,.1])
    parser.add_argument('-optLr', help='Find the optimal lr. (minLr, maxLr)', nargs=2, default=None)
    parser.add_argument('-heatMap', help='Show network in/outs as heatmap for n examples', type=int, default=0)
    parser.add_argument('-load', help="""Load a specific model. Defaults to latest model.
    If no latest model, will create a new one. If specified will load model of path input""", default='latest')

    # TODO: These need better UI's
    parser.add_argument('-trainFiles', help='Use files between (inp1,inp2) for training', type=int, nargs=2, default=[0,100])
    parser.add_argument('-valFiles', help='Use files between (inp1,inp2) for validation', type=int, nargs=2, default=[100,101])

    args = parser.parse_args()

    # Set default options if the differ
    maxEpochs = args.epochs

    return args

args = parseArgs()
trainNet(args)


#trainNet('SavedModels/GoNetLeaky_2_47_65_2.472.dnn', True)