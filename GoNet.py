from __future__ import print_function
import numpy as np
import cntk as cntk
from cntk.initializer import he_normal, normal
from cntk.layers import MaxPooling, BatchNormalization, Convolution, Dense, Dropout
from cntk.ops import relu

from Globals import BoardDepth, BoardLength, BoardLengthP, BoardSize, BoardSizeP

inputShape = (BoardDepth, BoardLength, BoardLength)


def Conv(input, filterShape, filters,  strides=(1,1), padding=False):
    c = Convolution(filterShape, filters, activation=None, pad=padding)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096)
    return relu(r)

def goNet(input, filters):

    conv = Conv(input, (5, 5), filters, strides=(2, 2))



goNet()


def trainNet():
    inputVar = cntk.ops.input_variable(inputShape, np.float32)
    labelVar = cntk.ops.input_variable(BoardSize, np.float32)