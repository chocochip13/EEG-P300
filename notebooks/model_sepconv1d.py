# Load Libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def SepConv1D(Chans=6, Samples=206, Filters=32):
    eeg_input    = Input(shape = (Samples, Chans))
    padded       = ZeroPadding1D(padding = 4)(eeg_input)
    block1       = SeparableConv1D(Filters, 16, strides = 8,
                                 padding = 'valid',
                                 data_format = 'channels_last',
                                 kernel_initializer = 'glorot_uniform',
                                 bias_initializer = 'zeros',
                                 use_bias = True)(padded)
    block1       = Activation('tanh')(block1)
    flatten      = Flatten(name = 'flatten')(block1)
    prediction   = Dense(1, activation = 'sigmoid')(flatten)

    return Model(inputs = eeg_input, outputs = prediction, name='SepConv1D')
