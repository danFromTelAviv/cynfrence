import time
import numpy as np
from convolutions import seperable_conv2d_no_padding as seperable_conv2d
from convolutions import conv2d_no_padding as conv2d


num_runs = 10
time_total = 0
for _ in range(num_runs):
    s = time.clock()
    a = conv2d(np.random.rand(5, 16, 16, 32), np.random.rand(3, 3, 32, 16), strides_height=2, dilation_height=2, dilation_width=2)
    time_total += time.clock() - s
# print(np.mean(a))
print("%d runs (conv2d) = %f (s)" % (num_runs, time_total))
# print(np.shape(a))


time_total = 0
for _ in range(num_runs):
    s = time.clock()
    b = seperable_conv2d(np.random.rand(5, 29, 29, 32), np.random.rand(3, 3, 32, 1), np.random.rand(1, 1, 32, 16), strides_height=2, strides_width=2)
    time_total += time.clock() - s
# print(np.mean(b))
print("%d runs (seperable_conv2d) = %f (s)" % (num_runs, time_total))
