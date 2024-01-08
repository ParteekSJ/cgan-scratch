import torch
from torch import nn
from torchvision import transforms

MNIST_SHAPE = (1, 28, 28)
N_CLASSES = 10
N_EPOCHS = 200
Z_DIM = 64
DISPLAY_STEP = 100
BATCH_SIZE = 128
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CRITERION = nn.BCEWithLogitsLoss()
TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)
DEBUG = False
LOG_PATH = "./logs"
BASE_DIR = "."
CHECKPOINT_DIR = "./checkpoints"
IMAGE_DIR = "./images"
