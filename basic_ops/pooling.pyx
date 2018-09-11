from __future__ import print_function
# %load_ext cython
import Cython
print(Cython.__version__)

# %%cython -a
cimport cython
import numpy as np
from convolutions import conv2d_no_padding, conv2d_with_padding

@cython.boundscheck(False)
@cython.wraparound(False)
def max_pooling_2d_no_padding(double[:, :, :, ::1] input_tensor, int kernel_height=1, int kernel_width=1):
    '''

    :param input_tensor: ( num_batches, height, width, channels )
    :param kernel_height: 1x1 int
    :param kernel_width: 1x1 int
    :return: output_tensor

    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    output[b, i, j, k] = strided max filter ( input )

    '''

    cdef Py_ssize_t input_batch_size = input_tensor.shape[0]
    cdef Py_ssize_t input_height = input_tensor.shape[1]
    cdef Py_ssize_t input_width = input_tensor.shape[2]
    cdef Py_ssize_t input_channels = input_tensor.shape[3]

    cdef Py_ssize_t kernel_height_reduction = (kernel_height-1)
    cdef Py_ssize_t kernel_width_reduction = (kernel_width-1)


    cdef Py_ssize_t output_height = int((input_height-kernel_height_reduction) / kernel_height)
    cdef Py_ssize_t output_width = int((input_width-kernel_width_reduction) / kernel_width)

    output = np.zeros((input_batch_size, output_height, output_width, input_channels), dtype=np.double)
    cdef double[:, :, : , :] output_view = output


    cdef int i, j, b, di, dj, q,strided_i, strided_j
    for i, strided_i in zip(range(output_height), range(0, output_height*kernel_height, kernel_height)):
        for j, strided_j in zip(range(output_width), range(0, output_width*kernel_width, kernel_width)):
            for di in range(kernel_height):
                for dj in range(kernel_width):
                    for b in range(input_batch_size):
                        for q in range(input_channels):
                            if ((di == 0 and dj == 0) or
                                output_view[b, i, j, q] < input_tensor[b, strided_i + di, strided_j + dj, q]):
                                output_view[b, i, j, q] = input_tensor[b, strided_i + di, strided_j + dj, q]
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def conv2d_with_padding(double[:, :, :, ::1] input_tensor, int kernel_height=1, int kernel_width=1):
    # reflects : https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
    cdef Py_ssize_t kernel_height_padding_top = int(kernel_height/2)
    cdef Py_ssize_t kernel_height_padding_bottom = int(kernel_height/2) + ((kernel_height+1) % 2)
    cdef Py_ssize_t kernel_width_padding_left = int(kernel_width/2)
    cdef Py_ssize_t kernel_width_padding_right = int(kernel_width/2) + ((kernel_width+1) % 2)
    input_tensor_padded = np.pad(input_tensor, ((0,0), (kernel_height_padding_top, kernel_height_padding_bottom),
                                                (kernel_width_padding_left, kernel_width_padding_right), (0,0)))
    return max_pooling_2d_no_padding(input_tensor_padded, kernel_height, kernel_width)

@cython.boundscheck(False)
@cython.wraparound(False)
def average_pooling_2d_no_padding(double[:, :, :, ::1] input_tensor, int kernel_height=1, int kernel_width=1):
    return conv2d_no_padding(input_tensor,
     np.ones((input_tensor.shape[0], kernel_height, kernel_width, input_tensor.shape[-1]), dtype=np.double)/
        (double(kernel_height)*double(kernel_width)),
                          strides_height=kernel_height, strides_width=kernel_width)


@cython.boundscheck(False)
@cython.wraparound(False)
def average_pooling_2d_with_padding(double[:, :, :, ::1] input_tensor, int kernel_height=1, int kernel_width=1):
    return conv2d_with_padding(input_tensor,
     np.ones((input_tensor.shape[0], kernel_height, kernel_width, input_tensor.shape[-1]), dtype=np.double)/
        (double(kernel_height)*double(kernel_width)),
                          strides_height=kernel_height, strides_width=kernel_width)
