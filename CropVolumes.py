import numpy as np

import utils

"""
This file contains functions that will return a randomly cropped arrays padded to the original dimension.
This is to increase the training examples
"""


def crop_array(array, start, dim):

    for i in range(len(start)):
        if start[i] + dim[i] > array.shape[i]:
            dim[i] = array.shape[i] - start[i]
    return array[start[0]: start[0] + dim[0], start[1]: start[1] + dim[1], start[2]: start[2] + dim[2], ]


def random_crops(array, n_crops):
    size = (len(array.shape), n_crops)
    print(size)
    start = np.random.randint(low=min(array.shape) // 2, size=size)
    dim = np.random.randint(max(array.shape) // 2, high=max(array.shape), size=size)
    print(start)
    return [
               crop_array(array, start[:, i], dim[:, i])
               for i in range(n_crops)
           ], array.shape


def crop_and_pad(array, n_crops):
    crop_list, size = random_crops(array, n_crops)
    print("crop list", crop_list)
    padded_crops = [utils.pad_array(crop, size) for crop in crop_list]
    return padded_crops
