- [ ] Add
- [ ] Subtract
- [ ] Mutiply
- [ ] Average
- [ ] Maximum
- [ ] Concatenate

def add(tensor_list):
    return sum(tensor_list)

def subtract(tensor_1, tensor_2):
    return tensor_1-tensor_2

def multiply(tensor_list):
    out_tensor = tensor_list[0]
    for new_tensor in tensor_list[1:]:
        out_tensor *= new_tensor
    return out_tensor

def average(tensor_list):
    return add(tensor_list) / len(tensor_list)

def maximum(tensor_list):
    out_tensor = tensor_list[0]
    for new_tensor in tensor_list[1:]:
        out_tensor = np.maximum(out_tensor, new_tensor)
    return out_tensor

def concatenate(tensor_list, axis=-1):
    return np.concatenate(tensor_list, axis=axis)
