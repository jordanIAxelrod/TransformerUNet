import TransformerUNet
import torch
import torch.nn as nn


def train_loop(model: nn.Module, train_dataloader, optim, loss_eq):
    total_loss = 0
    model.train()
    for data in train_dataloader:
        X, y = data[0], data[1]

        pred = model(X)

        loss = loss_eq(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.detach().item()

    return total_loss / len(train_dataloader)

def val_loop(model, val_dataloader, loss_eq):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            X, y = batch[0], batch[1]

            pred = model(X)

            loss = loss_eq(pred, y)

            total_loss += loss.detach().item()

    return total_loss / len(val_dataloader)


def fit(model, optim, loss_eq, train_dataloader, val_dataloader, epochs, url):

    train_loss, val_loss = [], []

    print("Training model")
    for epoch in range(epochs):
        print("_"*25, f"Epoch {epoch + 1}", "_"*25)

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

        torch.save(
            model,
            url + f" {epoch}.pt"
        )
        print(f"Training loss: {train_loss[epoch]:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

    return train_loss, val_loss
