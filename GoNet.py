from __future__ import print_function
import numpy as np
import cntk as cntk
from cntk.ops import relu
from cntk.initializer import he_normal, normal
from cntk.layers import MaxPooling, BatchNormalization, Dense, Dropout, Convolution2D
from DataGenerator import Generator
from Globals import BoardDepth, BoardLength, BoardLengthP, BoardSize, BoardSizeP
import scipy

#import cntk.tests.test_utils
#cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for CNTK internal build system)

batchSize = 512
maxEpochs = 100
featurePath = "./data/features"
labelPath = "./data/labels"

def Conv(input, filterShape, filters,  strides=(1,1), padding=False):
    c = Convolution2D(filterShape, filters, activation=None, pad=padding)(input)
    b = BatchNormalization(map_rank=1, normalization_time_constant=4096)(c)
    # Add dropout
    return relu(b)

def goNet(input, filters, outSize):
    
    # TODO: Look into what this padding turns out to actually pad to
    # See if we can get padding similar to AlphaGo's
    conv = Conv(input, (5, 5), filters, padding=True)
    c1  = Conv(conv, (3, 3), filters)
    c2 = Conv(c1, (3,3), filters, padding=True)

    pool = MaxPooling((2,2))(c2)
    z = Dense(outSize)(pool)

    return z

def trainNet():
    
    gen = Generator(featurePath, labelPath, (0, 4), batchSize)

    inputVar = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    labelVar = cntk.ops.input_variable(BoardSize, np.float32) #, dynamic_axes=input_dynamic_axes

    net = goNet(inputVar, 64, BoardSize)
   
    # Loss and metric
    loss = cntk.cross_entropy_with_softmax(net, labelVar)
    #loss = cntk.squared_error(net, labelVar)
    acc = cntk.classification_error(net, labelVar)
    
    minisPerBatch = gen.stepsPerEpoch()

    learner = cntk.adam(net.parameters, 0.0018, 0.9, minibatch_size=None)

    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)
    trainer = cntk.Trainer(net, (loss, acc), learner, progressPrinter)

    g = gen.generator()
    for epoch in range(maxEpochs):
        
        miniBatches = 0
        while miniBatches < minisPerBatch:
            X, Y = next(g)
            miniBatches += 1 # TODO: NEED to make sure this doesn't go over minibatchSize so we're not giving innacurate #'s

            trainer.train_minibatch({inputVar : X, labelVar : Y})

       
        trainer.summarize_training_progress()
            

        


trainNet()

