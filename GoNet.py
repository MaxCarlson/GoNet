from __future__     import print_function
import os
import re
import math
import glob
import numpy        as np
import cntk         as cntk
from Net            import goNet
from NetHeatMap     import NetHeatMap
from DataGenerator  import Generator
from argparse       import ArgumentParser
from misc           import printAccuracy, learningRateCycles, findOptLr
from Globals        import BoardDepth, BoardLength, BoardSize, BoardSizeP

batchSize           = 128
maxEpochs           = 50
defaultLr           = 0.01
defaultFileCount    = 598
resBlockCount       = 10
netFilters          = 64
# TODO: Command line args
featurePath = "./data/features"
labelPath   = "./data/labels"
saveDir     = './SavedModels/'
netName     = 'GoNet'

# TODO: This is a very ugly load system
# make it better!!
def findLatestModel(loadName):
    latestModel = loadName
    if loadName == 'latest':
        models = glob.glob(saveDir + '*.dnn')
        latestModel = max(models, key=os.path.getctime)
    elif loadName.lower() != 'new':
        latestModel = saveDir + loadName
    
    return latestModel

# TODO: possibly switch over to a checkpoint based load system
# as we crash enough to make it worth the saved progress benefit
#def findLatestCheckpoint():
#    chkpnts = glob.glob(saveDir + '*.chk')
#    latest  = max(chkpnts, key=os.path.getctime)
#    return latest

# TODO: Redo this load system, possibly switch to checkpoints,
# as it's super ugly!
def loadModel(args, inputVar, filters, resBlocks):
    epochOffset = 0
    net         = cntk.placeholder() 
    modelName   = findLatestModel(args.load)

    if modelName.lower() == 'new':
        net = goNet(inputVar, filters, resBlocks, BoardSize, 2)
        print('Created new network!')
    elif modelName != None:
        net         = cntk.load_model(modelName)
        reSearch    = re.search("\d+", modelName)
        epochOffset = int(modelName[reSearch.start():reSearch.end()])
        print('Sucessful load of model', modelName, '\n')

    return net, epochOffset

def trainNet(args): 

    # Crash doesn't seem to occur with this flag,
    # unfortunatly, it reduces training speed by about 35%
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Instantiate generators for both training and
    # validation datasets. Grab their generator functions
    # TODO: Command line args
    # TODO: Better system for using files testing/validation than ranges?
    tFileShp = (1,3)
    vFileShp = (0,1)
    gen      = Generator(featurePath, labelPath, tFileShp, batchSize, loadSize=3)
    valGen   = Generator(featurePath, labelPath, vFileShp, batchSize, loadSize=1)
    g        = gen.generator()
    vg       = valGen.generator()

    inputVar    = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), name='features')
    policyVar   = cntk.ops.input_variable((BoardSize))
    valueVar    = cntk.ops.input_variable((2)) 

    if args.fp16:
        cntk.cast(inputVar,  dtype=np.float16)
        cntk.cast(policyVar, dtype=np.float16)
        cntk.cast(valueVar,  dtype=np.float16)

    net, epochOffset = loadModel(args, inputVar, netFilters, resBlockCount)

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
    #error       = valueError
    error       = policyError

    if args.fp16:
        loss = cntk.cast(loss, dtype=np.float32)
        error = cntk.cast(error, dtype=np.float32)


    lrc = args.lr
    if args.cycleLr[0]:
        lrc = learningRateCycles(*args.cycleLr, gen.stepsPerEpoch, args.cycleMax)
        lrc = lrc * maxEpochs
    elif args.optLr:
        lrc = findOptLr(maxEpochs, *args.optLr, gen.stepsPerEpoch)

    lrc     = cntk.learners.learning_parameter_schedule(lrc, batchSize, batchSize)
    learner = cntk.adam(net.parameters, lrc, momentum=0.9, minibatch_size=batchSize, l2_regularization_weight=0.0001) 
    #learner = cntk.adadelta(net.parameters, lrc, l2_regularization_weight=0.0001) # Test adelta out!

    # TODO: Figure out how to write multiple 'metrics'
    tbWriter        = cntk.logging.TensorBoardProgressWriter(freq=1, log_dir='./TensorBoard/', model=net)
    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   
    trainer         = cntk.Trainer(net, (loss, error), learner, [progressPrinter, tbWriter])

    # TODO: Replace model load with loading/saving checkpoints!
    # So we can store learners state et al
    #trainer.restore_from_checkpoint(findLatestModel('latest'))
    #checkpointFreq = gen.stepsPerEpoch // checkpointFreq
    
    ls          = []
    losses      = []
    #valueAccs   = []
    #policyAccs  = []

    for epoch in range(maxEpochs):
        
        miniBatches = 0
        while miniBatches < gen.stepsPerEpoch:
            X, Y, W = next(g)
            miniBatches += 1 
            trainer.train_minibatch({ net.arguments[0] : X, policyVar : Y, valueVar : W }) 
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

    # TODO: Replace these globals with a config file reader!!!
    # As well as a config manager for multiple networks (so we can easily choose at start which net to boot from)
    # TODO: Auto saving start options to config file!
    global maxEpochs
    global defaultLr
    global netName
    global defaultFileCount
    global resBlockCount
    global netFilters

    parser.add_argument('-epochs',      help='Max # of epochs to train for', type=int, default=maxEpochs)
    parser.add_argument('-lr',          help='Set learning rate', type=float, default=defaultLr)
    parser.add_argument('-cycleLr',     help='Cycle learning rate between inp1-inp2, input 0 is cycle length', type=float, nargs=3, default=[0,.0,.0])
    parser.add_argument('-cycleMax',    help='Start the learning rate cycle at max instead of min', type=bool, default=False)
    parser.add_argument('-optLr',       help='Find the optimal lr. (minLr, maxLr)', nargs=2, default=None)
    parser.add_argument('-heatMap',     help='Show network in/outs as heatmap for n examples', type=int, default=0)
    parser.add_argument('-load',       help="""Load a specific model. Defaults to latest model.
    If no latest model, will create a new one. If specified will load model of path input""", default='latest')
    parser.add_argument('-name',        help='Change default name of the network', default=netName)
    parser.add_argument('-fileCount',   help='Set the number of files used for train+test data', type=int, default=defaultFileCount)
    parser.add_argument('-split',       help='Set the % split between train and validation data. (0.1 == 10% validation data)', type=float, default=0.1)
    parser.add_argument('-resBlocks',   help='Number of residual blocks for the new network to use', type=int, default=resBlockCount)
    parser.add_argument('-filters',     help='Set the number of convolutional filters for the new net to use', type=int, default=netFilters)
    parser.add_argument('-fp16',        help="Create a network using fp16 instead of fp32. Defaults to 0.", type=int, default=0)

    # TODO: Add in checkpoints as default storage method
    #parser.add_argument('-loadC',       help='Load the latest checkpoint if it exists', type=bool, default=False)
    #parser.add_argument('-checks',      help='How often through an epoch should we save checkpoints 11 =~ 11 checkpoints an epoch', type=int, default=checkpointFreq)

    args = parser.parse_args()

    # Set default options if they differ
    # TODO: Replace with config file
    maxEpochs           = args.epochs
    defaultLr           = args.lr
    netName             = args.name
    defaultFileCount    = args.fileCount
    resBlockCount       = args.resBlocks
    netFilters          = args.filters

    return args

args = parseArgs()
trainNet(args)