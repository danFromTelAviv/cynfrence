from __future__ import print_function
import Cython
print(Cython.__version__)

cimport cython
import numpy as np
from cython.parallel cimport prange

#cython: boundscheck=False, wraparound=False, nonecheck=False

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def conv2d_no_padding(double[:, :, :, ::1] input_tensor, double[:, :, :, ::1] kernels,
                          int dilation_height=1, int dilation_width=1,
                          int strides_height=1, int strides_width=1):
    '''

    :param input_tensor: ( num_batches, height, width, channels )
    :param kernels: ( height, width, channels, kernel_num ) channels_kernel == channels_input_tensor
    :param dilation_height: 1x1 int
    :param dilation_width: 1x1 int
    :param strides_height: 1x1 int
    :param strides_width: 1x1 int
    :return: output_tensor

    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    output[b, i, j, k] =
    sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                    filter[di, dj, q, k]

    '''
    cdef Py_ssize_t kernels_height = kernels.shape[0]
    cdef Py_ssize_t kernels_width = kernels.shape[1]
    cdef Py_ssize_t kernels_channels = kernels.shape[2]
    cdef Py_ssize_t kernels_num_filters = kernels.shape[3]
    cdef Py_ssize_t kernel_height_reduction = (kernels_height-1)
    cdef Py_ssize_t kernel_width_reduction = (kernels_width-1)
    cdef Py_ssize_t input_batch_size = input_tensor.shape[0]
    cdef Py_ssize_t input_height = input_tensor.shape[1]
    cdef Py_ssize_t input_width = input_tensor.shape[2]
    cdef Py_ssize_t input_channels = input_tensor.shape[3]

    assert kernels_channels == input_channels

    cdef Py_ssize_t output_height = int((input_height-kernel_height_reduction) / strides_height)
    cdef Py_ssize_t output_width = int((input_width-kernel_width_reduction) / strides_width)

    cdef double[:, :, : , :] output_view = np.zeros((input_batch_size, output_height, output_width, kernels_num_filters), dtype=np.double)
    # cdef double[:, :, : , :] output_view = output


    cdef Py_ssize_t  i, j, b, k, di, dj, q
    cdef int dilated_i, dilated_j, strided_i, strided_j, di_strided_dilated, dj_strided_dilated
    #for i in prange(output_height, nogil=True):
    for i in range(output_height):
        strided_i = i*strides_height
        for j  in range(output_width):
            strided_j = j*strides_width
            for di  in range(kernels_height):
                di_strided_dilated = di*dilation_height+strided_i
                for dj in range(kernels_width):
                    dj_strided_dilated = dj*dilation_width+strided_j
                    for b in range(input_batch_size):
                        for k in range(kernels_num_filters):
                            for q in range(kernels_channels):
                                output_view[b, i, j, k] += \
                                    input_tensor[b, di_strided_dilated, dj_strided_dilated, q]  \
                                                      * kernels[di, dj, q, k]
    return output_view

@cython.boundscheck(False)
@cython.wraparound(False)
def conv2d_with_padding(double[:, :, :, ::1] input_tensor, double[:, :, :, ::1] kernels,
                          int dilation_height=1, int dilation_width=1,
                          int strides_height=1, int strides_width=1):
    # reflects : https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
    cdef Py_ssize_t kernel_height_padding_top = int(kernels.shape[0]/2)
    cdef Py_ssize_t kernel_height_padding_bottom = int(kernels.shape[0]/2) + ((kernels.shape[0]+1) % 2)
    cdef Py_ssize_t kernel_width_padding_left = int(kernels.shape[1]/2)
    cdef Py_ssize_t kernel_width_padding_right = int(kernels.shape[1]/2) + ((kernels.shape[0]+1) % 2)
    input_tensor_padded = np.pad(input_tensor, ((0,0), (kernel_height_padding_top, kernel_height_padding_bottom),
                                                (kernel_width_padding_left, kernel_width_padding_right), (0,0)))
    return conv2d_no_padding(input_tensor_padded, kernels,
                             dilation_height, dilation_width, strides_height, strides_width)

@cython.boundscheck(False)
@cython.wraparound(False)
def seperable_conv2d_no_padding(double[:, :, :, ::1] input_tensor,
                            double[:, :, :, ::1] kernels_depthwise, double[:, :, :, ::1] kernel_pointwise,
                            int dilation_height=1, int dilation_width=1,
                            int strides_height=1, int strides_width=1):
    '''

    :param input_tensor: ( num_batches, height, width, channels )
    :param kernels_depthwise: ( height, width, channels, kernel_num ) channels_kernel == channels_input_tensor
    :param kernel_pointwise: ( height, width, channels, kernel_num ) channels_kernel == channels_input_tensor
    :param dilation_height: 1x1 int
    :param dilation_width: 1x1 int
    :param strides_height: 1x1 int
    :param strides_width: 1x1 int
    :return: output_tensor

    https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d
    output[b, i, j, k] = sum_{di, dj, q, r}
    input[b, strides[1] * i + di, strides[2] * j + dj, q] *
    depthwise_filter[di, dj, q, r] *
    pointwise_filter[0, 0, q * channel_multiplier + r, k]

    '''

    cdef Py_ssize_t input_batch_size = input_tensor.shape[0]
    cdef Py_ssize_t input_height = input_tensor.shape[1]
    cdef Py_ssize_t input_width = input_tensor.shape[2]
    # input_channels = input_tensor.shape[3]
    # kernel_pointwise_height = kernel_pointwise.shape[0] #== 1
    # kernel_pointwise_num_width = kernel_pointwise.shape[1] #== 1
    cdef Py_ssize_t kernel_pointwise_channels = kernel_pointwise.shape[2] # == 1
    cdef Py_ssize_t kernel_pointwise_num_filers = kernel_pointwise.shape[3] #
    cdef Py_ssize_t kernels_depthwise_height = kernels_depthwise.shape[0]
    cdef Py_ssize_t kernels_depthwise_width = kernels_depthwise.shape[1]
    cdef Py_ssize_t kernels_depthwise_channels = kernels_depthwise.shape[2]
    cdef Py_ssize_t kernels_depthwise_num_filters = kernels_depthwise.shape[3]

    cdef Py_ssize_t kernel_height_reduction = (kernels_depthwise_height-1)
    cdef Py_ssize_t kernel_width_reduction = (kernels_depthwise_width-1)
    cdef Py_ssize_t output_height = int((input_height-kernel_height_reduction) / strides_height)
    cdef Py_ssize_t output_width = int((input_width-kernel_width_reduction) / strides_width)


    output = np.zeros((input_batch_size, output_height, output_width, kernel_pointwise_num_filers), dtype=np.double)
    cdef double[:, :, : , :] output_view = output

    cdef Py_ssize_t channel_multiplier = kernels_depthwise_num_filters
    assert channel_multiplier*kernels_depthwise_channels == kernel_pointwise_channels

    cdef int i, j, b, k, di, dj, q, r
    # num_ops = 0
    cdef int dilated_i, dilated_j, strided_i, strided_j, di_dilated, dj_dilated, q_x_multiplier
    for i, strided_i in zip(range(output_height), range(0,output_height*strides_height, strides_height)):
        for j, strided_j in zip(range(output_width), range(0,output_width*strides_width, strides_width)):
            for di, di_dilated in zip(range(kernels_depthwise_height), range(0, kernels_depthwise_height*dilation_height, dilation_height)):
                for dj, dj_dilated in zip(range(kernels_depthwise_width), range(0, kernels_depthwise_width*dilation_width, dilation_width)):
                    for q, q_x_multiplier in zip(range(kernels_depthwise_channels), range(0, kernels_depthwise_channels * channel_multiplier, channel_multiplier)):
                        for b in range(input_batch_size):
                            for k in range(kernel_pointwise_num_filers):
                                for r in range(kernels_depthwise_num_filters):
                                    output_view[b, i, j, k] += \
                                        input_tensor[b, strided_i + di_dilated, strided_j + dj_dilated, q] \
                                                          * kernels_depthwise[di, dj, q, r] * kernel_pointwise[0, 0, q_x_multiplier + r, k]
    #                                 num_ops +=1
    # print("ops_breakdown: ")
    # print(i, j, b, k, di, dj, q, r)
    # print("num_ops :", num_ops)
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def seperable_conv2d_with_padding(double[:, :, :, ::1] input_tensor,
                            double[:, :, :, ::1] kernels_depthwise, double[:, :, :, ::1] kernel_pointwise,
                            int dilation_height=1, int dilation_width=1,
                            int strides_height=1, int strides_width=1):
    # reflects : https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
    cdef Py_ssize_t kernel_height_padding_top = int(kernels_depthwise.shape[0]/2)
    cdef Py_ssize_t kernel_height_padding_bottom = int(kernels_depthwise.shape[0]/2) + ((kernels_depthwise.shape[0]+1) % 2)
    cdef Py_ssize_t kernel_width_padding_left = int(kernels_depthwise.shape[1]/2)
    cdef Py_ssize_t kernel_width_padding_right = int(kernels_depthwise.shape[1]/2) + ((kernels_depthwise.shape[0]+1) % 2)
    input_tensor_padded = np.pad(input_tensor, ((0,0), (kernel_height_padding_top, kernel_height_padding_bottom),
                                                (kernel_width_padding_left, kernel_width_padding_right), (0,0)))
    return seperable_conv2d_no_padding(input_tensor_padded, kernels_depthwise, kernel_pointwise,
                                       dilation_height, dilation_width, strides_height, strides_width)
