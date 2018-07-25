from __future__ import print_function
import os
import re
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
    epochOffset = 0

    if modelName != None:
        net = cntk.load_model(modelName)
        epochOffset = int(modelName[re.search("\d", modelName).start()])
        print('Sucessful load of model', modelName, '\n')
    else:
        net = goNet(inputVar, filters, BoardSize, 2)
        print('Created new network!')

    return net, epochOffset

def trainNet(args): 
    
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

    net, epochOffset = loadModel(args)

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
        lrc = learningRateCycles(*args.cycleLr, gen.stepsPerEpoch)
        lrc = lrc * maxEpochs
    elif args.optLr:
        lrc = findOptLr(maxEpochs, *args.optLr, gen.stepsPerEpoch)

    lrc = cntk.learners.learning_parameter_schedule(lrc, batchSize, batchSize)
    learner = cntk.adam(net.parameters, lrc, momentum=0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) 

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

        net.save(saveDir + netName + '_{}_{}_{}_{:.3f}.dnn'.format(epoch+1+epochOffset, policyAcc, valueAcc, losses[epoch][1]))

def parseArgs():
    parser = ArgumentParser()

    # TOP TODO: Auto load latest saved model
    global maxEpochs
    global defaultLr
    global netName

    parser.add_argument('-epochs',  help='Max # of epochs to train for', type=int, default=maxEpochs)
    parser.add_argument('-lr',      help='Set learning rate', type=float, default=defaultLr)
    parser.add_argument('-cycleLr', help='Cycle learning rate between inp1-inp2, input 0 is cycle length', nargs=3, default=[0,.0,.0])
    parser.add_argument('-optLr',   help='Find the optimal lr. (minLr, maxLr)', nargs=2, default=None)
    parser.add_argument('-heatMap', help='Show network in/outs as heatmap for n examples', type=int, default=0)
    parser.add_argument('-load',    help="""Load a specific model. Defaults to latest model.
    If no latest model, will create a new one. If specified will load model of path input""", default='latest')
    parser.add_argument('-name', help='Change default name of the network', default=netName)

    # TODO: These need better UI's
    # TODO: What is a better way to pick train/test files automatically by size?
    # Auto split with input % test vs % train. smallest # files as validation data so we won't run into
    # previously trained on data if we increase data input size
    parser.add_argument('-trainFiles',  help='Use files between (inp1,inp2) for training', type=int, nargs=2, default=[0,100])
    parser.add_argument('-valFiles',    help='Use files between (inp1,inp2) for validation', type=int, nargs=2, default=[100,101])

    args = parser.parse_args()

    # Set default options if the differ
    maxEpochs = args.epochs
    defaultLr = args.lr
    netName   = args.name

    return args

args = parseArgs()
trainNet(args)


#trainNet('SavedModels/GoNetLeaky_2_47_65_2.472.dnn', True)