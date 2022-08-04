import numpy as np
from PIL import Image
import os
import pandas as pd
import utils

"""
This file contains functions to read in the MRI slices and save them as volumes

Functions
----------
stack_images: np.array
    Takes a list of MRI slices and stacks them as multiple pictures in an array

read_mri: list
    Takes a folder name and reads in all slices into a list

read_all_mri: list
    Takes the root folder and reads in all MRI


"""


def stack_images(img_list: list, dim: tuple) -> np.array:
    return np.array(img_list).reshape(dim)


def read_mri(folder: str, dim_df: pd.DataFrame, sub_folder: str) -> list:
    slice_list = []
    files = os.listdir(folder)
    for file in files:
        with Image.open(folder + '/' + file) as im:
            slice_list.append(np.array(im))
    name = sub_folder + '_' + file[:10]
    row = dim_df.loc[dim_df['Case Name'] == name, ['Height', 'Width']].values
    h, w = row[0][0], row[0][1]
    slice_list = stack_images(slice_list, (h, w, len(files)))
    return slice_list


def read_all_mri(folder: str, dim_df) -> list:
    folders = os.listdir(folder)
    mri_list = []
    mri_names = []
    for fold in folders:
        for sub_fold in os.listdir(fold):
            im_folder = fold + '/' + sub_fold + r'\scans'
            print(im_folder)
            mri_names.append(sub_fold)
            mri_list.append(read_mri(im_folder, dim_df, sub_fold))

    return mri_list, mri_names


def stack_all_mri(mri_list: list) -> list:
    print(mri_list[0][0])
    return [stack_images(ls) for ls in mri_list]


def main(folder: str) -> list:
    dim_df = pd.read_excel(r"../Image Dimensions.xlsx")

    mri_list, mri_names = read_all_mri(folder, dim_df)

    h, w, d = dim_df['Height'].max(), \
              dim_df['Width'].max(), \
              dim_df['Case Name'].apply(lambda x: x[:13]).value_counts()[0]
    temp_list = []
    for mri in mri_list:
        temp_list.append(utils.pad_array(mri, (h, w, d)))
    mri_list = temp_list
    print(mri_names[0])
    return mri_list, mri_names
