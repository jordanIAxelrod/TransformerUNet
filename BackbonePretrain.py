import TransformerUNet
import torch
import torch.nn as nn
import Dataset
from Config import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import Train_model
import segmentation_models_pytorch as smp


class Head(nn.Module):
    def __init__(self, backbone, depth, activation, n_classes):
        super(Head, self).__init__()
        self.backbone = backbone
        self.input = nn.Linear(32 * 32 * CHAN_LIST[-1], 32 * 32 * CHAN_LIST[-1])
        self.hidden = nn.ModuleList(
            [
                nn.Linear(32 * 32 * CHAN_LIST[-1], 32 * 32 * CHAN_LIST[-1] )
                for _ in range(depth)
            ]
        )
        self.out = nn.Linear(32 * 32 * CHAN_LIST[-1] * 5, n_classes)
        self.activation = activation()
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)[0]
        x = self.input(x.flatten(1))
        for layer in self.hidden:
            x = layer(self.activation(x))
        return self.sofmax(self.out(self.activation(x)))


def main():
    print(DEVICE)
    net_backbone = TransformerUNet.Backbone(
        img_size=IMG_SIZE,
        n_patches_down=N_PATCHES_DOWN,
        n_patches_up=N_PATCHES_UP,
        depth=DEPTH,
        chan_list=CHAN_LIST,
        mlp_ratio=MLP_RATIO,
        qkv_bias=QKV_BIAS,
        n_layers=N_LAYERS,
        p=P,
        attn_p=ATTN_P
    )
    net_backbone.to(DEVICE)
    net = Head(net_backbone, 2, nn.ReLU, 1000)
    net.to(DEVICE)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Rotate(limit=35, p=.1),
        A.VerticalFlip(p=.2),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    data = Dataset.ImagenetClassification(
        "../inputs/imagenet images/ILSVRC/Data/CLS-LOC/train",
        "../inputs/LOC_synset_mapping.txt",
        transform
    )
    train_len = int(len(data) * 0.8)
    test_len = len(data) - train_len
    train_data, test_data = torch.utils.data.random_split(data, [train_len, test_len])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=BATCH_SIZE * 4,
        num_workers=NUM_WORKERS * 2
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

    Train_model.fit(net, optimizer, loss, train_loader, test_loader, NUM_EPOCHS, "outputs/pretrain/")


if __name__ == "__main__":
    main()
