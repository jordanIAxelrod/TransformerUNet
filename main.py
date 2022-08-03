import TransformerUNet
import Train_model
import torch.optim as optim
import torch.nn.modules.loss as loss
import CropVolumes

import DataLoader

DATA_URL = ""
LABEL_URL = ""


def clean():
    pass


def fit_model():
    model = input(
        "Please select the parameters of the model in the following order separated my commas enter nothing if using "
        "default params: \nImage size, \nn_patches_down, \nn_patches_up,\nchan_list,\nmlp_ratio,\nqkv_bias,"
        "\nn_layers,\ndropout, \nattn_dropout")
    model = model.split(", ")
    img_size, n_patches_down, n_patches_up, chan_list, mlp_ratio, qkv_bias,n_layers, p, attn_p = model
    img_size = int(img_size)
    n_patches_down = int(n_patches_down)
    n_patches_up = int(n_patches_up)
    chan_list = [int(chan) for chan in chan_list.split(", ")]
    mlp_ratio = float(mlp_ratio)
    qkv_bias = bool(qkv_bias)
    n_layers = int(n_layers)
    p = float(p)
    attn_p = float(attn_p)
    model = TransformerUNet.UNetViT(
        img_size=img_size,
        n_patches_down=n_patches_down,
        n_patches_up=n_patches_up,
        chan_list=chan_list,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        n_layers=n_layers,
        p=p,
        attn_p=attn_p
    )
    opt = input("What optimizer shall be used")
    if opt.lower() == "adam":
        opt = optim.Adam(model.parameters(), lr=.0001)
    else:
        opt = optim.SGD(model.parameters(),lr=.0001)

    loss_eq = loss.BCELoss()

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
