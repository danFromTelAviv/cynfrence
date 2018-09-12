from __future__ import print_function
# %load_ext cython
import Cython
print(Cython.__version__)

# %%cython -a
cimport cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def relu(input_tensor, double threshold=0):
    return np.maximum(input_tensor, threshold)

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid(input_tensor, double threshold=0):
    return np.expit(input_tensor)

@cython.boundscheck(False)
@cython.wraparound(False)
def tanh(input_tensor):
    return np.tanh(input_tensor)


@cython.boundscheck(False)
@cython.wraparound(False)
def softmax(x):
    e_x = np.exp(x - np.max(x,axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@cython.boundscheck(False)
@cython.wraparound(False)
def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    y = np.minimum(y, 1.)
    y = np.maximum(y, 0.)
    return y
