"""
CNN1 model implementation for EEG signal classification.

References:
    - Cecotti et al. 2011: https://ieeexplore.ieee.org/document/5492691
    - Lecun 1989: http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Input, Conv1D, Activation, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import logging

logger = logging.getLogger(__name__)

def cecotti_normal(shape, dtype=None, partition_info=None):
    """
    Initializer proposed by Cecotti et al. 2011.
    
    Parameters
    ----------
    shape : tuple
        Shape of the tensor to initialize
    dtype : dtype
        Data type of the tensor
    partition_info : dict
        Additional information for partitioned variables
        
    Returns
    -------
    tensor
        Initialized tensor
    """
    if len(shape) == 1:
        fan_in = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
    else:
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size

    return K.random_normal(shape, mean=0.0, stddev=(1.0 / fan_in))


def scaled_tanh(z):
    """
    Scaled hyperbolic tangent activation function.
    
    As proposed by LeCun 1989 and described in LeCun et al. 1998.
    
    Parameters
    ----------
    z : tensor
        Input tensor
        
    Returns
    -------
    tensor
        Activated tensor
    """
    return 1.7159 * K.tanh((2.0 / 3.0) * z)


# Register custom activation function
get_custom_objects().update({'scaled_tanh': Activation(scaled_tanh)})


def CNN1(Chans=6, Samples=206):
    """
    CNN1 model for EEG classification.
    
    Parameters
    ----------
    Chans : int
        Number of EEG channels
    Samples : int
        Number of time samples
        
    Returns
    -------
    keras.models.Model
        CNN1 model
    """
    logger.info(f"Creating CNN1 model with {Chans} channels and {Samples} samples")
    
    eeg_input = Input(shape=(Samples, Chans))

    block1 = Conv1D(10, 1, padding='same',
                   data_format='channels_last',
                   bias_initializer=cecotti_normal,
                   kernel_initializer=cecotti_normal,
                   use_bias=True)(eeg_input)
    block1 = Activation('scaled_tanh')(block1)

    block1 = Conv1D(50, 13, padding='same',
                   data_format='channels_last',
                   bias_initializer=cecotti_normal,
                   kernel_initializer=cecotti_normal,
                   use_bias=True)(block1)
    block1 = Activation('scaled_tanh')(block1)

    flatten = Flatten(name='flatten')(block1)
    dense = Dense(100, activation='sigmoid')(flatten)
    prediction = Dense(2, activation='sigmoid')(dense)

    return Model(inputs=eeg_input, outputs=prediction, name='CNN1')
