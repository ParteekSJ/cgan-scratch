import os
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import ipdb

from model.generator import Generator, create_noise_vector
from model.discriminator import Discriminator
from constants import *
from utils.utils import (
    plot_images_from_tensor,
    weights_init,
    ohe_vector_from_labels,
    concat_vectors,
    calculate_input_dim,
    init_setting,
)
from logger.logger import setup_logging, get_logger


if __name__ == "__main__":
    os.environ["IPDB_CONTEXT_SIZE"] = "7"
    exp_path, checkpoint_dir, image_dir = init_setting()
    setup_logging(save_dir=exp_path)

    logger = get_logger(name="train")  # log message printing

    dataloader = DataLoader(
        dataset=MNIST(root="data", download=False, transform=TRANSFORMS),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    gen_input_dim, disc_input_chan = calculate_input_dim(
        z_dim=Z_DIM, mnist_shape=MNIST_SHAPE, n_classes=N_CLASSES
    )  # (64+10=74, 10+1=11)

    gen = Generator(input_dim=gen_input_dim).to(DEVICE)
    gen_opt = torch.optim.Adam(params=gen.parameters(), lr=LR)
    disc = Discriminator(im_chan=disc_input_chan).to(DEVICE)
    disc_opt = torch.optim.Adam(params=disc.parameters(), lr=LR)

    gen, disc = gen.apply(weights_init), disc.apply(weights_init)

    generator_losses, discriminator_losses = [], []
    cur_step = 0

    # ipdb.set_trace()
    for epoch in range(N_EPOCHS):
        for idx, (images, labels) in enumerate(dataloader):
            cur_batch_size = len(images)  # 128
            images = images.to(DEVICE)

            """
            Create OHE vectors from labels (ground truth), i.e., 
            - labels[0] = 8
            - one_hot_labels[0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] 
            """

            one_hot_labels = ohe_vector_from_labels(
                label_tensor=labels, n_classes=N_CLASSES  # [128, 10]
            ).to(DEVICE)
            image_one_hot_labels = one_hot_labels[..., None, None]  # [128, 10, 1, 1]

            # [128, 10, 1, 1] -> # [128, 10, 28, 28]
            image_one_hot_labels = image_one_hot_labels.repeat(
                1, 1, MNIST_SHAPE[1], MNIST_SHAPE[2]
            )  # how many times to repeat each dim

            ### Train Discriminator
            disc_opt.zero_grad()
            fake_noise = create_noise_vector(
                n_samples=cur_batch_size, input_dim=Z_DIM, device=DEVICE
            )  # [128, 64]

            """
            IMPORTANT:
            * For Generator, labels are appened to the end of the noise vectors.
            * For Discriminator, labels are appended to the channel dimension of the fake & real images (dim=1).
            """

            # z(noise) - [128, 64] + y(true_labels) - [128, 10]
            noise_and_labels = concat_vectors(fake_noise, one_hot_labels)  # [128, 74]

            # noise_and_labels dims get expanded automatically during generator's forward pass
            fake = gen(noise_and_labels)  # CONDITIONED FAKE IMAGES / [128, 1, 28, 28]

            # [128, 1, 28, 28] + [128, 10, 28, 28] = [128, 11, 28, 28] (both)
            fake_image_and_labels = concat_vectors(fake, image_one_hot_labels)
            real_image_and_labels = concat_vectors(images, image_one_hot_labels)

            # Getting the discriminator's predictions
            disc_fake_pred = disc(fake_image_and_labels.detach())  # [128, 1]
            disc_real_pred = disc(real_image_and_labels)  # [128, 1]

            # Calculating the Discriminator's Loss
            disc_fake_loss = CRITERION(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = CRITERION(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            # Backpropagate & Update Weights
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Keep track of average discriminator losses.
            discriminator_losses += [disc_loss.item()]

            ### Train Generator
            gen_opt.zero_grad()

            # [128, 1, 28, 28] + [128, 10, 28, 28] -> [128, 11, 28, 28]
            fake_image_and_labels = concat_vectors(fake, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)  # [128, 1]

            gen_loss = CRITERION(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of average generator losses.
            generator_losses += [gen_loss.item()]

            if idx % DISPLAY_STEP == 0 and idx > 0:
                # Calculate Generator Mean Loss for the latest display steps (i.e., last 50 steps)
                gen_mean = sum(generator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                disc_mean = sum(discriminator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                logger.info(
                    f"Epoch {epoch}: | Step: {idx} | Gen Loss: {gen_mean} | Disc Loss: {disc_mean}"
                )

            cur_step += 1

        checkpoint = {
            "epoch": epoch,
            "gen_state_dict": gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "gen_optimizer": gen_opt.state_dict(),
            "disc_optimizer": disc_opt.state_dict(),
        }  # save state dictionary
        torch.save(checkpoint, f"{checkpoint_dir}/model.pth")

        plot_images_from_tensor(fake, plot_name=f"{image_dir}/epoch-{epoch}-fake.png")
        plot_images_from_tensor(images, plot_name=f"{image_dir}/epoch-{epoch}-real.png")
