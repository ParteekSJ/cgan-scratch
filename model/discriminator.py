import sys

sys.path.append("../")
import torch
from torch import nn
from torchinfo import summary
import ipdb
import os
from constants import *


class Discriminator(nn.Module):
    def __init__(self, im_chan: int = 1, hidden_dim: int = 64):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            self._discriminator_block(im_chan, hidden_dim),
            self._discriminator_block(hidden_dim, hidden_dim * 2),
            self._discriminator_block(hidden_dim * 2, 1, final_layer=True),
        )

    def _discriminator_block(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        final_layer: int = False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        if DEBUG:
            ipdb.set_trace()
        disc_pred = self.discriminator(image)
        return disc_pred.view(len(disc_pred), -1)


"""
QUESTIONS
1. Why do we have to downsample as we're moving through the network?
2. Why does the kernel size remain 4 across the network?
"""

if __name__ == "__main__":
    os.environ["IPDB_CONTEXT_SIZE"] = "7"

    ipdb.set_trace()

    disc = Discriminator(im_chan=1, hidden_dim=64)
    sample_image = torch.randn(10, 1, 28, 28)
    disc_pred = disc(sample_image)
    print(f"{disc_pred.shape=}")  # [10, 1]

    print(summary(disc, input_data=sample_image))

    """
    Summary of the Generator's Forward Pass.
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Discriminator                            [10, 1, 1, 1]             --
    ├─Sequential: 1-1                        [10, 1, 1, 1]             --
    │    └─Sequential: 2-1                   [10, 64, 13, 13]          --
    │    │    └─Conv2d: 3-1                  [10, 64, 13, 13]          1,088
    │    │    └─BatchNorm2d: 3-2             [10, 64, 13, 13]          128
    │    │    └─LeakyReLU: 3-3               [10, 64, 13, 13]          --
    │    └─Sequential: 2-2                   [10, 128, 5, 5]           --
    │    │    └─Conv2d: 3-4                  [10, 128, 5, 5]           131,200
    │    │    └─BatchNorm2d: 3-5             [10, 128, 5, 5]           256
    │    │    └─LeakyReLU: 3-6               [10, 128, 5, 5]           --
    │    └─Sequential: 2-3                   [10, 1, 1, 1]             --
    │    │    └─Conv2d: 3-7                  [10, 1, 1, 1]             2,049
    ==========================================================================================
    """
