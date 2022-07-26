import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Split images into patches and then embed them.

    Parameters
    ----------
    img_size : in
        Size of the image (it is a square).

    patch_size : int
        Size of the patch it is a square

    in_chans : int
        Number of input channels.

    embed_dim : int
        The emmbedding dimension

    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # no overlap convolutional layer
        assert embed_dim == in_chans * self.patch_size ** 2, \
            f"Embed dim must be {in_chans * self.patch_size ** 2}"
        self.patch = nn.conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass

        :param x: torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)
        :return: torch.Tensor
            Shape `(n_smaples, n_patches, embed_dim)
        """

        x = self.patch(x)  # (n_smaples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed, n_patches)
        return x.transpose(1, 2)


class Attention(nn.Module):
    """
    Attention Mechanism

    Parameters
    ----------
    dim : int
        The input and the output dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability appl;ied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------
    scale : float
        Normalizing constant for the dot product
    qkv: nn.linear
        Linear projection for the query, key, value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all
        the attention heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0., proj_p: float = 0.):
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass
        :param x: torch.Tensor
             Shape `(n_smaples, n_patches + 1, dim)`.
        :return: torch.Tensor
             Shape `(n_smaples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError('Incorrect dimension')
        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)

        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ).permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)

        dp = (
                     q @ k_t
             ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches + 1, head_dim)

        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)

        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Parameters
    ---------
    in_features : int
        Number of input features
    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
         Number of output features

    p : float
        Dropout probability

    n_layers : int
        number of hidden layers

    Attribute
    ---------
    fc : nn.Sequential
        the hidden layers

    act : nn.GELU
        GELU activation function

    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, p: float = 0.,
                 n_layers: int = 1) -> None:
        super().__init__()
        mod_list = []
        mod_list.append(nn.Linear(in_features, hidden_features))
        mod_list.append(nn.GELU())
        for _ in range(n_layers - 1):
            mod_list.append(nn.Linear(hidden_features, hidden_features))
            mod_list.append(nn.GELU())
        self.output = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        self.mlp = nn.sequential(mod_list)

    def forward(self, x):
        """
        Rund forward pass.
        :param x:  torch.Tensor
            Shape: `(n_samples, n_patches + 1, in_features)`.
        :return: torch.Tensor
            Shape: `(n_samples, n_patches + 1, out_features'.
        """

        x = self.mlp(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)
        x = self.output(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)
        return x


class Block(nn.Module):
    """
    Transformer block.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the 'MLP' module with respect to 'dim'
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    n_layers : int
        number of layers in the MLP
    out_features : int
        Number of output features.
    p, attn_p: float
        Dropout probability

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attnetion Attention module.

    mlp: MLP
        MLP module.
    """

    def __init__(
            self,
            dim: int,
            n_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            n_layers: int = 1,
            out_features: int = None,
            p: float = 0.,

            attn_p: float = .0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, esp=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )

        self.norm2 = nn.LayerNorm(dim, esp=1e-6)
        hidden_features = int(dim * mlp_ratio)
        if not out_features:
            out_features = dim
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=out_features,
            n_layers=n_layers,
        )

    def forward(self, x) -> torch.Tensor:
        """
        Run forward pass
        :param x: torch.Tensor
            Shape: `(n_samples, n_patches + 1, dim)`.
        :return: torch.Tensor
            Shape: `(n_samples, n_patches  + 1, out_features)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class ViT(nn.Module):
    """
    Simplified implementation of the Vision Transformer.

    Parameters
    ----------
    img_size : int
        Bothe height and the width of the image (it is a square).
    patch_size : int
        both height and the width of the patch (it is a square).
    in_chans: int
        Number of channels.
    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    n_layers : int
        Number of layers int he `MLP` module
    p, attn_p : float
        Dropout probability

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence. It has `embed_dim` elements.
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches. It has `(n_pathces + 1) * embed_dim` elements.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization
    """

    def __init__(
            self,
            img_size: int = 384,
            patch_size: int = 16,
            in_chans: int = 3,
            n_classes: int = 1000,
            embed_dim: int = 768,
            depth: int = 12,
            n_heads: int = 12,
            mlp_ratio: float = 4,
            qkv_bias: bool = True,
            n_layers: int = 1,
            p: float = 0.,
            attn_p: float = 0.
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    n_layers=n_layers,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, esp=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass
        :param x: torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        :return: torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """

        n_samples = x.shape[0]

        x = self.patch_embed(x)
        cls = self.cls_token.expand(
            n_samples, -1, -1
        )
        x = torch.cat((cls, x), dim=1)  # (n_samples, n_patches + 1, embed_dim)
        x = x + self.pos_emb  # (n_samples, n_patches + 1, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]

        x = self.head(cls_token_final)
        return x


class Convolutions(nn.Module):
    """
    Layers of convolutions with the same input and output channels

    """

    def __init__(self, in_chans, n_chans: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, n_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does the forward pass of Convolutions
        :param x: torch.Tensor
            Shape: `(n_samples, n_chans, height, width)'
        :return: torch.Tensor
            Shape: `(n_samples, n_chans, height - 4, width - 4)'
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNetViT(nn.Module):
    """
    Implementation of UNet using transformers instead of CNN

    Parameters
    ----------
    img_size : int
        Bothe height and the width of the image (it is a square).
    patch_size : int
        both height and the width of the patch (it is a square).
    in_chans: int
        Number of channels.
    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    n_layers : int
        Number of layers int he `MLP` module
    p, attn_p : float
        Dropout probability

    Attributes
    ----------
    down : nn.ModuleList
        list of blocks of transformers on the down side of the network
    up : nn.ModuleList
        List of blocks of transformers on the up side of the network
    bottleneck : nn.ModuleList
        List of blocks of transformers making up the bottle neck
    skip_cons : list
        List of tensors of skip connections.
    """

    def __init__(
            self,
            img_size: int = 512,
            n_patches_down: int = 16,
            n_patches_up: int = 64,
            in_chans: int = 3,
            chan_list: list = (3, 16, 32, 64, 128, 256),
            n_heads: int = 12,
            mlp_ratio: float = 4,
            qkv_bias: bool = True,
            n_layers: int = 1,
            p: float = 0.,
            attn_p: float = 0.
    ):
        super(UNetViT, self).__init__()
        self.n_patches_down = n_patches_down
        self.n_patches_up = n_patches_up
        self.dims = [img_size // 2 ** i for i in range(4)]
        self.down_patch_embed = nn.ModuleList(
            [
                PatchEmbed(
                    img_size=dim,
                    patch_size=dim // self.n_patches_down ** .5,  # n_chan * img_size // n_patches
                    in_chans=chan_list[i],
                    embed_dim=chan_list[i] * dim ** 2 // self.n_patches_down
                )
                for i, dim in enumerate(self.dims)
            ]
        )

        self.pos_emb_down = [
            nn.Parameter(
                torch.zeros(1, self.n_patches_down, chan_list[i] * dim ** 2 // self.n_patches_down)
            )
            for i, dim in enumerate(self.dim)
        ]

        self.pos_drop = nn.Dropout(p=p)

        self.down = nn.ModuleList(
            [
                Block(
                    dim=dim ** 2 * chan_list[i] // self.n_patches_down,
                    n_heads=chan_list[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    n_layers=n_layers,
                    p=p,
                    attn_p=attn_p,
                )
                for i, dim in enumerate(self.dims)
            ]
        )

        # The convolutional section for the encoder and the bottleneck
        self.down_conv = nn.ModuleList(
            [
                Convolutions(chan, chan_list[i + 1])
                for i, chan in enumerate(chan_list[:-1])
            ]
        )
        self.max_pool = nn.MaxPooling2d(2)

        # Patch embedding for the decoder
        self.up_patch_embed = nn.ModuleList(
            [
                PatchEmbed(
                    img_size=self.dims[i],
                    patch_size=self.dims[i] // self.n_patches_up ** .5,
                    in_chans=chan_list[i + 1],
                    embed_dim=chan_list[i + 1] * self.dims[i] ** 2 // self.n_patches_up
                )
                for i in range(len(self.dims) - 1, -1, -1)
            ]
        )
        self.up = nn.ModuleList(
            [
                Block(
                    dim=self.dims[i] ** 2 * chan_list[i+1] // self.n_patches_up,
                    n_heads=chan_list[i+1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    n_layers=n_layers,
                    p=p,
                    attn_p=attn_p,
                )
                for i in range(len(self.dims) - 1, -1, -1)
            ]
        )
        self.up_conv = nn.ModuleList(
            [
                Convolutions(chan_list[i], chan_list[i - 1])
                for i in range(len(self.chan_list) - 1, 1, -1)
            ]
        )

        self.convtranspose2d = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    chan_list[i],
                    chan_list[i - 1],
                    kernel_size=2,
                    stride=2
                )
                for i in range(len(chan_list) - 1, 1, -1)
            ]
        )

    def inverse_patch(self, x: torch.Tensor, n_chans) -> torch.Tensor:
        """

        :param x: torch.Tensor
            Shape: `(n_samples, n_patches, embed_dim)`
        :param n_chans: int
            number of channels
        :return: torch.Tensor
            Shape: `(n_samples, 1, H, W)
        """
        n_samples, n_patches, embed_dim = x.shape
        patch_size = (embed_dim // n_chans) ** .5
        x = x.reshape(-1, n_patches, n_chans, patch_size, patch_size)
        # (n_samples, n_patches, H, W)
        rows = []
        for j in n_patches ** .5:
            z = x[:, j * n_patches: (j + 1) * n_patches]
            rows.append(torch.cat(
                [z[:, k].reshape(n_samples, n_chans, patch_size, patch_size)
                 for k in range(z.shape[1])],
                dim=3))
            # (n_samples, n_patches ** .5, h, w)
        x = torch.cat(rows, dim=4)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer UNet
        :param x: torch.Tensor
            Shape: `(n_samples, n_chans, img_size, img_size)`
        :return: torch.Tensor
            Shape: `(n_samples, 1, img_size, img_size)`
        """
        skip_connections = []

        # Downward pass
        for i, block in enumerate(self.down):
            x = self.down_patch_embed[i](x) + self.pos_emb_down[i]  # (n_samples, n_patches, dim1)
            x = block(x)  # (n_samples, n_patches, dim1)
            x = self.inverse_patch(x)  # (n_samples, n_chans, H, W)
            x = self.down_conv[i](x)  # (n_samples, n_chans, H, W)
            skip_connections.append(x)
            x = self.max_pool(x)  # (n_samples, n_chans, (H - 1) / 2, (W - 1) / 2)

        # Bottleneck
        x = self.bottle_neck(x)
        skip_connections.reverse()
        self.pos_emb.reverse()
        for i, skip_connection in enumerate():
            x = self.convtranspose2d[i](x)
            x = torch.cat([x,skip_connection], dim=1) # (n_samples, n_chans * 2, H, W)
            x = self.up_patch_embed[len(skip_connections) - i - 1](x) + self.pos_emb_up[i]
            x = self.up(x) # (n_samples, n_patches * 2
            x = self.inverse_patch(x)
            x = self.up_conv[i](x)

