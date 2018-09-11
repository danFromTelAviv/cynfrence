from __future__ import print_function
# %load_ext cython
import Cython
print(Cython.__version__)

# %%cython -a
cimport cython
import numpy as np


def batch_normalization(x, mean, variance, offset, scale):
    # refer to https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
    return scale*(x-mean)/variance + offset
