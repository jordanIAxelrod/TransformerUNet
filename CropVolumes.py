import numpy as np
from Config import *
import utils

"""
This file contains functions that will return a randomly cropped arrays padded to the original dimension.
This is to increase the training examples
"""


def crop_array(array: np.array, start: list, dim: list, twoD: bool):
    """
    Crops the array given the dim
    :param array: np.array
    shape (h, w, c) or (h, w, d, c)
    :param start: list
    start of the crop
    :param dim: list
    dimensions of the crop
    :param twoD: bool
    whether the crop is a volume or image
    :return:
    """
    for i in range(len(start)):
        if start[i] + dim[i] > array.shape[i]:
            dim[i] = array.shape[i] - start[i]
    if not twoD:

        return array[start[0]: start[0] + dim[0], start[1]: start[1] + dim[1], start[2]: start[2] + dim[2], ]
    else:
        return array[start[0]: start[0] + dim[0], start[1]: start[1] + dim[1], ]


def random_crops(array, n_crops, twoD):
    size = (len(array.shape) - 1, n_crops)
    start = np.random.randint(low=int(min(array.shape) * crop_px), size=size)
    dim = np.random.randint(int(max(array.shape) * crop_px), high=max(array.shape), size=size)
    return [
               crop_array(array, start[:, i], dim[:, i], twoD)
               for i in range(n_crops)
           ], array.shape


def crop_and_pad(array, n_crops, twoD=True):
    crop_list, size = random_crops(array, n_crops, twoD)
    padded_crops = [utils.pad_array(crop, size) for crop in crop_list]
    return padded_crops
