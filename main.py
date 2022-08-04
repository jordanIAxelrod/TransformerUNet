import os

import TransformerUNet
import Train_model
import torch.optim as optim
import torch.nn.modules.loss as loss

import albumentations as A
from albumentations.pytorch import ToTensorV2
from Config import *
import CropVolumes

import Dataset
from sklearn.model_selection import train_test_split

DATA_URL = ""
LABEL_URL = ""


def clean():
    labels = os.listdir(LABEL_URL + "/Raw")
    data = os.listdir(DATA_URL + "/Raw")


def get_inputs():
    model = input(
        "Please select the parameters of the model in the following order separated my commas enter nothing if using "
        "default params: \nImage size, \nn_patches_down, \nn_patches_up,\nchan_list,\nmlp_ratio,\nqkv_bias,"
        "\nn_layers,\ndropout, \nattn_dropout")
    model = model.split(", ")
    img_size_local, n_patches_down, n_patches_up, chan_list, mlp_ratio, qkv_bias, n_layers, p, attn_p = model
    img_size_local = int(img_size_local)
    n_patches_down = int(n_patches_down)
    n_patches_up = int(n_patches_up)
    chan_list = [int(chan) for chan in chan_list.split(", ")]
    mlp_ratio = float(mlp_ratio)
    qkv_bias = bool(qkv_bias)
    n_layers = int(n_layers)
    p = float(p)
    attn_p = float(attn_p)
    epochs = int(input("How many epochs to train for"))

    return {"image_size": img_size_local,
            'n_patches_down': n_patches_down,
            'n_patches_up': n_patches_up,
            "chan_list":chan_list,
            "mlp_ratio": mlp_ratio,
            "qkv_bias":qkv_bias,
            "n_layers":n_layers,
            "p":p,
            "attn_p":attn_p}, epochs


def fit_model():
    model_params, epochs = get_inputs()
    model = TransformerUNet.UNetViT(
        *model_params
    )
    opt = input("What optimizer shall be used")
    if opt.lower() == "adam":
        opt = optim.Adam(model.parameters(), lr=.0001)
    else:
        opt = optim.SGD(model.parameters(), lr=.0001)

    loss_eq = loss.BCELoss()
    data = [data for data in os.listdir(DATA_URL + "/clean")]
    labels = [label for label in os.listdir(LABEL_URL + "/clean")]

    train_transform = A.Compose([
        A.Resize(img_size),
        A.Rotate(limit=35, p=1),
        A.VerticalFlip(p=.2),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(img_size)
    ])

    train_X, train_y, test_X, test_y = train_test_split(data, labels, test_size=0.1)
    train_dataloader = Dataset.ImageDataLoader(train_y, train_X, transforms=train_transform)
    val_dataloader = Dataset.ImageDataLoader(test_y, test_X, transforms=val_transform)

    Train_model.fit(model, opt, loss_eq, train_dataloader, val_dataloader, epochs)


def main():
    use = input("Clean Data, Fit Model or Both. (1,2, or 3)")
    if use == 1:
        clean()

    elif use == 2:
        fit_model()

    else:
        clean()
        fit_model()


if __name__ == "__main__":
    main()
