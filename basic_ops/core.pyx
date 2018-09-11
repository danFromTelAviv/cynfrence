from __future__ import print_function
# %load_ext cython
import Cython
print(Cython.__version__)

# %%cython -a
cimport cython
import numpy as np

# for activation - use the activation functions directly.

def dense( double[:,::1] input_tensor, double[:,::1] weights, double[::1] biases ):
    return weights * input_tensor + biases


def flatten(input_tensor):
    return input_tensor.reshape((input_tensor.shape[0],-1))


def reshape(input_tensor, target_shape):
    '''
    :param input_tensor: nd arrayd
    :param target_shape: tuple representing the requested dimentions other than the batch dimention ( 0 )
    :return: output_tensor reshaped to target_shape

    for example
    input_tensor = np.random.rand(5,10,10,3)
    target_shape = (300)
    would result in a matrix of shape ( 5, 300 )
    '''
    target_shape = (input_tensor.shape[0],) + target_shape
    return input_tensor.reshape((target_shape)




