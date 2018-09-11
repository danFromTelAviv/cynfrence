import numpy as np
from core import dense, reshape

input_tensor = np.ones((3, 15))
weights = np.random.rand(3, 15)
biases = np.random.rand(15)
out = dense(input_tensor, weights, biases)
assert(out.shape == input_tensor.shape)

input_tensor = np.random.rand(5, 10, 10, 3)
target_shape = (300)
out = reshape(input_tensor, target_shape)
assert(len(out.shape[:]) == 2)