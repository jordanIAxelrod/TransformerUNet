import numpy as np


def pad_array(ary: np.array, max_dim: tuple):
    if len(ary.shape) != len(max_dim):
        raise ValueError('Max dimension shapes must match the shape of the passed array')

    padding = []
    for i in range(len(ary.shape)):
        dif = max_dim[i] - ary.shape[i]
        if dif % 2 == 0:
            padding.append([dif / 2, dif / 2])
        else:
            padding.append([dif // 2 + 1, dif // 2])
    return np.pad(ary, padding)
