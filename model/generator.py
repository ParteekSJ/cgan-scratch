import sys

sys.path.append("../")
import torch
from torch import nn
from torchinfo import summary
import ipdb
import os
from constants import *

# 1, Since we're using ConvTranspose2d [upsampling], we want to downsample during the forward pass of the GENERATOR.
# 2, Kernel Size remains 4 across the network .
# 3, Consider the MNIST example, i.e., y is between 0 to 9. We encode the label value to a one-hot vector. For instance, 3 will be encoded as
# (0,0,1,0,0,0,0,0,0). This LABEL vector along with noise (z) is fed to the generator G to create an image that resembles "3".
# For the discriminator, we add the supposed label as one-hot encoded vector to its input, i.e., label vector, true image, and generated image.


class Generator(nn.Module):
    def __init__(self, input_dim: int = 10, im_chan: int = 1, hidden_dim: int = 64):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.generator = nn.Sequential(
            self._generator_block(input_dim, hidden_dim * 4),  # <- upsampling
            self._generator_block(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1
            ),
            self._generator_block(hidden_dim * 2, hidden_dim),
            self._generator_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def _generator_block(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        final_layer: int = False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.Tanh(),
            )

    def forward(self, noise):
        if DEBUG:
            ipdb.set_trace()
        # 2D -> 4D, i.e., adding H & W dimensions for ConvTranspose2d op
        noise = noise.view(len(noise), self.input_dim, 1, 1)  # [B, C, H, W]
        return self.generator(noise)


def create_noise_vector(n_samples: int, input_dim: int, device: str = "cpu"):
    return torch.randn(n_samples, input_dim).to(device)


if __name__ == "__main__":
    os.environ["IPDB_CONTEXT_SIZE"] = "7"

    ipdb.set_trace()
    noise = create_noise_vector(n_samples=100, input_dim=16)  # [100, 16]

    # [100, 16] -> (reshape) -> [100, 16, 1, 1] -> (gen) -> [100, 1, 28, 28]
    gen = Generator(input_dim=16, im_chan=1, hidden_dim=64)

    output = gen(noise)
    print(summary(gen, input_data=noise.view(len(noise), 16, 1, 1)))

    """
    Summary of the Generator's Forward Pass.
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Generator                                [100, 1, 28, 28]          --
    ├─Sequential: 1-1                        [100, 1, 28, 28]          --
    │    └─Sequential: 2-1                   [100, 256, 3, 3]          --
    │    │    └─ConvTranspose2d: 3-1         [100, 256, 3, 3]          37,120
    │    │    └─BatchNorm2d: 3-2             [100, 256, 3, 3]          512
    │    │    └─ReLU: 3-3                    [100, 256, 3, 3]          --
    │    └─Sequential: 2-2                   [100, 128, 6, 6]          --
    │    │    └─ConvTranspose2d: 3-4         [100, 128, 6, 6]          524,416
    │    │    └─BatchNorm2d: 3-5             [100, 128, 6, 6]          256
    │    │    └─ReLU: 3-6                    [100, 128, 6, 6]          --
    │    └─Sequential: 2-3                   [100, 64, 13, 13]         --
    │    │    └─ConvTranspose2d: 3-7         [100, 64, 13, 13]         73,792
    │    │    └─BatchNorm2d: 3-8             [100, 64, 13, 13]         128
    │    │    └─ReLU: 3-9                    [100, 64, 13, 13]         --
    │    └─Sequential: 2-4                   [100, 1, 28, 28]          --
    │    │    └─ConvTranspose2d: 3-10        [100, 1, 28, 28]          1,025
    │    │    └─Tanh: 3-11                   [100, 1, 28, 28]          --
    ==========================================================================================
    """
