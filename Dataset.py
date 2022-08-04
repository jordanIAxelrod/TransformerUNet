import os
from PIL import Image
import numpy as np


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

    def __init__(self, label_dir, data_dir, transforms=None):
        # Enumerate the labels and images in a list
        # Sort them so the idx of each are the same
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)
        self.transforms = transforms

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
        img = os.path.join(self.data_dir, img)
        msk = os.path.join(self.label_dir, img.replace(".jpg", "_mask.jgp"))

        img = np.array(Image.open(img).convert("RBG"))
        msk = np.array(Image.open(msk).convert("L"), dtype=np.float32)
        msk[msk == 255.] = 1
        if self.transforms:
            augmentations = self.transforms(image=img, mask=msk)
            img = augmentations["image"]
            msk = augmentations["mask"]
        return torch.Tensor(img), torch.Tensor(msk)