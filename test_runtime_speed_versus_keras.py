import time
import numpy as np
# from convolutions import seperable_conv2d_no_padding as seperable_conv2d
from basic_ops.convolutions import conv2d_no_padding as conv2d
# import keras
from keras import Input, Model
from keras.initializers import RandomUniform
from keras.layers import Conv2D
from keras.optimizers import Adam


def basic_conv_model():
    input_layer = Input(shape=(None, None, 32))
    conv = Conv2D(16, (3, 3), strides=(2,2), kernel_initializer=RandomUniform(), use_bias=False, padding="valid")(input_layer)
    model = Model(input_layer, conv)
    model.compile(Adam(), loss="mse")
    return model

model = basic_conv_model()
num_runs = 1000
time_total_cython = 0
time_total_tf = 0

for _ in range(num_runs):
    # input_tensor = np.random.rand(5, 51, 51, 32)
    input_tensor = np.random.rand(5, 16, 16, 32)
    weights = np.random.rand(3, 3, 32, 16)

    s_2 = time.clock()
    b = model.predict(input_tensor)
    time_total_tf += time.clock() - s_2

    s_1 = time.clock()
    # a = conv2d(input_tensor, weights,  dilation_height=2, dilation_width=2)
    a = conv2d(input_tensor, weights, strides_height=2, strides_width=2)
    time_total_cython += time.clock() - s_1



print(a.shape)
print(b.shape)

# print(np.mean(a))
print("%d runs (conv2d) = %f (s) - cython" % (num_runs, time_total_cython))
print("%d runs (conv2d) = %f (s) - tf" % (num_runs, time_total_tf))
