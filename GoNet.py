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
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096)(c)
    # Add dropout
    return relu(r)

def goNet(input, filters, outSize):
    
    conv = Conv(input, (5, 5), filters, padding=True)
    c1  = Conv(conv, (3, 3), filters)
    c2 = Conv(c1, (3,3), filters, padding=True)

    pool = MaxPooling((2,2))(c2)
    z = Dense(outSize)(pool)

    return z

def trainNet():
    
    gen = Generator(featurePath, labelPath, (0, 10), batchSize)

    inputVar = cntk.ops.input_variable((BoardDepth, BoardLength, BoardLength), np.float32, name='features')
    labelVar = cntk.ops.input_variable(BoardSize, np.float32) #, dynamic_axes=input_dynamic_axes

    net = goNet(inputVar, 64, BoardSize)
   
    # Loss and metric
    ce = cntk.cross_entropy_with_softmax(net, labelVar)
    pe = cntk.classification_error(net, labelVar)

    minibatchSize = gen.stepsPerEpoch()

    learner = cntk.adam(net.parameters, 0.0018, 0.9, minibatch_size=minibatchSize)

    progressPrinter = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)
    trainer = cntk.Trainer(net, (ce, pe), learner, progressPrinter)

    g = gen.generator()
    for epoch in range(maxEpochs):
        sampleCount = 0

        while sampleCount < batchSize:
            X, Y = next(g)
            sampleCount += minibatchSize # TODO: NEED to make sure this doesn't go over minibatchSize so we're not giving innacurate #'s

            #train_summary = ce.train((X, Y), parameter_learners=[learner], callbacks=[progressPrinter])
            trainer.train_minibatch({inputVar : X, labelVar : Y})
       
        trainer.summarize_training_progress()
            

        


trainNet()

