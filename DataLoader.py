import os
from PIL import Image
from numpy import asarray

import torch
from torch.utils.data import Dataset


def get_number(x):
    return int(x.split("/")[-1][:3])


class ImageDataLoader(Dataset):
    """
    Loads cleaned images and the masks for the model

    label_url: str
        the folder where the cleaned masks are saved
    data_url: str
        the folder where the cleaned images are stored
    """

    def __init__(self, label_url, data_url):
        # Enumerate the labels and images in a list
        # Sort them so the idx of each are the same
        self.labels = [label for label in os.listdir(label_url)]
        self.labels.sort(key=get_number)
        self.data = [data for data in os.listdir(data_url)]
        self.data.sort(key=get_number)

    def __len__(self):
        """
        gets the length of the list
        :return: number of images in the training set
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Gets the set of image and mask from the index
        :param idx: int
            the index
        :return: tuple
            tensor of masks and data
        """
        img = self.data[idx]
        msk = self.labels[idx]

        img = asarray(Image.open(img))
        msk = asarray(Image.open(msk))

        return torch.Tensor(img), torch.Tensor(msk)
