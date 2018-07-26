import cntk
from cntk.ops import relu
from cntk.layers import MaxPooling, BatchNormalization, Dense, Dropout, Convolution2D


def Conv(input, filterShape, filters, activation=True, padding=True):
    cn = Convolution2D(filterShape, filters, activation=None, pad=padding)(input)
    ba = BatchNormalization()(cn) 
    return relu(ba) if activation else ba

# Two convolutional layers with a skip
# connection from input to the seconds layers output
def ResLayer(input, filters):
    c0 = Conv(input, (3,3), filters)
    c1 = Conv(c0,    (3,3), filters, activation=False)
    return relu(c1 + input)

def ResStack(input, filters, count):
    inp = input
    for l in range(count):
        inp = ResLayer(inp, filters)
    return inp

# TODO: Possibly apply a softmax to this as well as ValueHead
def PolicyHead(input, pOutSize):
    pc = Conv(input, (1,1), 2, 1)
    return Dense(pOutSize, activation=cntk.softmax)(pc) 

# One Head of the network for predicting whether
# an input will result in a win for side to move or not
def ValueHead(input, size, vOutSize):
    vc = Convolution2D((1,1), 1, activation=None)(input)
    b0 = BatchNormalization()(vc)
    #dr = Dropout(0.14)(b0)
    r0 = relu(b0)
    d0 = Dense(size, activation=None)(r0)
    #do = Dropout(0.14)(d0)
    r1 = relu(d0)
    d1 = Dense(vOutSize, activation=None)(r1)
    return cntk.softmax(d1)

# TODO: Command line args
def goNet(input, filters, policyOut, valueOut):

    c0 = Conv(input, (3,3), filters) 
    rs = ResStack(c0, filters, 8)

    # TODO: Look into heads output softmax!
    p  = PolicyHead(rs, policyOut)
    v  = ValueHead(rs, 128, valueOut)
    
    return cntk.combine(p, v)
