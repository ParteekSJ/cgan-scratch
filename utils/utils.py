# Importing Modules
import sys
import os
import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
import ipdb
import datetime
from pathlib import Path
from constants import *

sys.path.append("../")
torch.manual_seed(0)


def plot_images_from_tensor(
    image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=False, plot_name=""
):
    image_tensor = (image_tensor + 1) / 2  # [-1, 1] -> [0, 1] (normalizes image)

    # detach a tensor from the current computational graph and moving it to CPU.
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)

    # [C, H, W] -> [H, W, C] (format expected by matplotlib)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    plt.savefig(f"{plot_name}")


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, val=0)


def ohe_vector_from_labels(label_tensor, n_classes):
    # takes 'label_tensor' tensor of shape (*), and converts it to 0/1s tensor of shape (*, n_classes)
    return F.one_hot(label_tensor, num_classes=n_classes)


"""
x = torch.tensor([4, 3, 2, 1, 0])
F.one_hot(x, num_classes=6)

# Expected result
# tensor([[0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0]])
"""


def concat_vectors(x, y):
    # Generator in CGAN doesn't only take the noise vector 'z' but also the label vector 'y'.
    # Hence, the CONCATENATION, i.e., Generator Input  - Noise + Label Vector
    combined = torch.cat(tensors=(x.float(), y.float()), axis=1)
    return combined


""" 
Concatenation of Multiple Tensor with `torch.cat()`
RULE - To concatenate WITH torch.cat(), where the list of tensors are concatenated across the specified dimensions, requires 2 conditions to be satisfied
    1. All tensors need to have the same number of dimensions, and
    2. All dimensions EXCEPT the one that they are concatenated on, need to have the same size. 

Concatenation between (32,2,32) and (32,4,32) with concat(dim=1) yields (32,6,32). 
"""


def calculate_input_dim(z_dim, mnist_shape, n_classes):
    """
    DISCRIMINATOR -> Class information is appended as a channel or some other method,
    GENERATOR -> Class information is encoded by appending a one-hot vector to the noise to form a long vector input.

    z_dim = size of the noise vector - 64 or 128,
    mnist_shape = (1, 28, 28)
    n_classes = 10 [mnist digits]

    """
    generator_input_dim = z_dim + n_classes  # latent noise [z] + label vector [y]

    discriminator_image_channel = (
        mnist_shape[0] + n_classes
    )  # label information is appended as a channel.

    return generator_input_dim, discriminator_image_channel


if __name__ == "__main__":
    os.environ["IPDB_CONTEXT_SIZE"] = "7"
    ipdb.set_trace()

    label_tensor = torch.arange(start=0, end=5)
    num_classes = len(torch.unique(label_tensor))
    ohe_vec = ohe_vector_from_labels(label_tensor, num_classes)

    """
    ONE HOT ENCODED VECTOR for [0, 1, 2, 3, 4]
    tensor([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]])
    """

    z_dim, mnist_shape = 64, (1, 28, 28)
    gen_inp_dim, disc_inp_dim = calculate_input_dim(z_dim, mnist_shape, num_classes)
    print(f"{gen_inp_dim=} | {disc_inp_dim=}")


def init_setting():
    timestr = str(datetime.datetime.now().strftime("%Y-%m%d_%H%M"))
    experiment_dir = Path(LOG_PATH)
    experiment_dir.mkdir(exist_ok=True)  # directory for saving experimental results
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)  # root directory of each experiment

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_dir = checkpoint_dir.joinpath(timestr)
    checkpoint_dir.mkdir(exist_ok=True)  # root directory of each checkpoint

    image_dir = Path(IMAGE_DIR)
    image_dir.mkdir(exist_ok=True)
    image_dir = image_dir.joinpath(timestr)
    image_dir.mkdir(exist_ok=True)  # root directory of each image (generated and real)

    # returns several directory paths
    return experiment_dir, checkpoint_dir, image_dir
