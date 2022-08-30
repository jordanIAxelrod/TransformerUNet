import os

import pandas as pd

import Clean
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
import segmentation_models_pytorch as smp


DATA_URL = ""
LABEL_URL = ""


def clean():
    labels = pd.read_csv(LABEL_URL)
    labels["rle"] = labels["rle"].apply(Clean.rle2mask)
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

    loss_eq = smp.losses.DiceLoss(
        "binary",
        from_logits=True,
        log_loss=True
    )
    data = [data for data in os.listdir(DATA_URL + "/clean")]
    labels = [label for label in os.listdir(LABEL_URL + "/clean")]

    train_transform = A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size),
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.transforms.ColorJitter(p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=.5),
            A.PiecewiseAffine(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.HueSaturationValue(10, 15, 10),
            A.CLAHE(clip_limit=4),
            A.RandomBrightnessContrast(),
        ], p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize()
    ])

    train_X, train_y, test_X, test_y = train_test_split(data, labels, test_size=0.1)
    train_dataset = Dataset.ImageDataLoader(train_y, train_X, transforms=train_transform)
    val_dataset = Dataset.ImageDataLoader(test_y, test_X, transforms=val_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE * 4,
        num_workers=NUM_WORKERS * 2
    )
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
