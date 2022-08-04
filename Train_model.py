import TransformerUNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.figure as figure

import tqdm
import datetime as dt
import numpy as np
from Config import *


def train_loop(model: nn.Module, train_dataloader: DataLoader, optim: torch.optim, loss_eq: nn.Module):
    """
    Runs the training loop for the epoch
    :param model: nn.Module
    the model were training
    :param train_dataloader: DataLoader
    the training dataloader
    :param optim: torch.optim
    the optimizer were using
    :param loss_eq: nn.Module
    The loss of the model
    :return: float
    the average error of the epoch
    """
    total_loss = 0
    model.train()
    loop = tqdm(train_dataloader)
    for batch_idx, (X, y) in enumerate(loop):
        X = X.to(device=DEVICE)
        y = y.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            pred = model(X)

            loss = loss_eq(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.detach().item()

        loop.set_postfix(loss=loss.detach().item())
    return total_loss / len(train_dataloader)


def val_loop(model: nn.Module, val_dataloader: DataLoader, loss_eq: nn.Module):
    """
    Runs the validation loop of training
    :param model: nn.Module
    the model were training
    :param val_dataloader: DataLoader
    the validation dataloader
    :param loss_eq: nn.Module
    The loss of the model
    :return: float
    the loss
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            X, y = batch[0], batch[1]

            pred = model(X)

            loss = loss_eq(pred, y)

            total_loss += loss.detach().item()

    return total_loss / len(val_dataloader)


def plot_loss(train_loss: float, val_loss: float, train_line: figure.Figure, val_line: figure.Figure):
    """
    Updates the plot of the losses
    :param train_loss: float
    loss of the training in this epoch
    :param val_loss: float
    loss of the validation in the epoch
    :param train_line: figure.Figure
    the line of the train loss
    :param val_line: figure.Figure
    the line of the validation loss
    :return: None
    """
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim([-1, max([ymax, train_loss, val_loss]) + 2])
    t_line = train_line.get_xdata()
    if len(t_line) == 0:
        x = 1
    else:
        x = t_line[-1] + 1
    train_line.set_data(np.append(t_line, [x]), np.append(train_line.get_ydata(), [train_loss]))

    val_line.set_data(np.append(val_line.get_xdata()), [x], np.append(val_line.get_ydata(), [val_loss]))

    plt.draw()


# TODO Early stopping algo
def fit(
        model: nn.Module,
        optim: torch.optim,
        loss_eq: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        url: str
):
    """
    Fits the model for the specified amount of epochs and saves iterations of the model
    :param model: nn.Module
    The model to fit
    :param optim: torch.optim
    the optimizer of the model
    :param loss_eq: nn.Module
    the loss of the network
    :param train_dataloader: DataLoader
    dataloader of training data
    :param val_dataloader: DataLoader
    dataloader of the validation data
    :param epochs: int
    number of epochs to train for
    :param url: str
    the location to store the model
    :return: tuple
    the validation and training losses for each epoch
    """
    train_loss, val_loss = [], []
    train_line, = plt.plot([], [], color="black", label="Training loss")
    val_line, = plt.plot([], [], color="red", label="Validation loss")
    plt.legend()
    plt.xlim([0, epochs])
    print("Training model")
    for epoch in range(epochs):
        try:
            print("_" * 25, f"Epoch {epoch + 1}", "_" * 25)

            train_loss.append(
                train_loop(
                    model,
                    train_dataloader,
                    optim,
                    loss_eq
                )
            )

            val_loss.append(
                val_loop(
                    model,
                    val_dataloader,
                    loss_eq
                )
            )
            fmt = "%m.%d.%Y"
            torch.save(
                model,
                url + f"{dt.datetime.now().strftime(fmt)} Transformer UNet {epoch}.pt"
            )
            print(f"Training loss: {train_loss[epoch]:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            plot_loss(train_loss[-1], val_loss[-1], train_line, val_line)
            plt.savefig(url + f"{dt.datetime.now().srtftime(fmt)} Training and Validation Loss.png")

            if all(val > val_loss[-patients - 1] - err for val in val_loss[-patients:]):
                print("Model has hit early stopping condition")
                break
        except KeyboardInterrupt as ki:
            time = dt.datetime.now().strftime("%m.%d.%Y")
            torch.save(
                model,
                url + f"{time} Transformer UNet Keyboard Interrupt {epoch}"
            )
            plt.savefig(url + f"{dt.datetime.now().srtftime(fmt)} Training and Validation Loss.png")

            raise ki

    return train_loss, val_loss
