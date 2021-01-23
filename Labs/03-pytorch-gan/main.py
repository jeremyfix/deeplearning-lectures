#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import argparse
import logging
import sys
import os
# External imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import deepcs.display
from deepcs.fileutils import generate_unique_logpath
import tqdm
# Local imports
import data
import models


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    # Parameters
    dataset = args.dataset
    dataset_root = args.dataset_root
    nthreads = args.nthreads
    batch_size = args.batch_size
    dropout = args.dropout
    debug = args.debug
    base_lr = args.base_lr
    num_epochs = args.num_epochs
    discriminator_base_c = args.discriminator_base_c
    generator_base_c = args.generator_base_c
    latent_size = args.latent_size
    sample_nrows = 8
    sample_ncols = 8

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Dataloaders
    train_loader, valid_loader, img_shape = data.get_dataloaders(dataset_root=dataset_root,
                                                                 cuda=use_cuda,
                                                                 batch_size=batch_size,
                                                                 n_threads = nthreads,
                                                                 dataset=dataset,
                                                                small_experiment=debug)

    # Model definition
    model = models.GAN(img_shape,
                       dropout,
                       discriminator_base_c,
                       latent_size,
                       generator_base_c)
    model.to(device)

    # Optimizers
    critic = model.discriminator
    generator = model.generator
    ######################
    # START CODING HERE ##
    ######################
    # Step 1 - Define the optimizer for the critic
    optim_critic = None
    # Step 2 - Define the optimizer for the generator
    optim_generator = None

    # Step 3 - Define the loss (it must embed the sigmoid)
    loss = None

    ####################
    # END CODING HERE ##
    ####################

    # Callbacks
    summary_text = "## Summary of the model architecture\n" + \
            f"{deepcs.display.torch_summarize(model)}\n"
    summary_text += "\n\n## Executed command :\n" +\
        "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    logdir = generate_unique_logpath('./logs', 'gan')
    tensorboard_writer = SummaryWriter(log_dir = logdir,
                                       flush_secs=5)
    tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

    with open(os.path.join(logdir, "summary.txt"), 'w') as f:
        f.write(summary_text)

    save_path = os.path.join(logdir, 'generator.pt')

    logger.info(f">>>>> Results saved in {logdir}")

    # Define a fixed noise used for sampling
    fixed_noise = torch.randn(sample_nrows*sample_ncols,
                              latent_size).to(device)

    # Generate few samples from the initial generator
    model.eval()
    fake_images = model.generator(X=fixed_noise)
    grid = torchvision.utils.make_grid(fake_images,
                                       nrow=sample_nrows,
                                       normalize=True)
    tensorboard_writer.add_image("Generated", grid, 0)
    torchvision.utils.save_image(grid, 'images/images-0000.png')

    # Training loop
    for e in range(num_epochs):

        tot_closs = tot_gloss = 0
        critic_accuracy = 0
        Nc = Ng = 0
        model.train()
        for ei, (X, _) in enumerate(tqdm.tqdm(train_loader)):

            # X is a batch of real data
            X = X.to(device)
            bi = X.shape[0]

            pos_labels = torch.ones((bi, )).to(device)
            neg_labels = torch.zeros((bi, )).to(device)

            ######################
            # START CODING HERE ##
            ######################
            # Step 1 - Forward pass for training the discriminator
            real_logits, _ = None
            fake_logits, _ = None

            # Step 2 - Compute the loss of the critic
            Dloss = None + None

            # Step 3 - Reinitialize the gradient accumulator of the critic
            None

            # Step 4 - Perform the backward pass on the loss
            None

            # Step 5 - Update the parameters of the critic
            None

            ####################
            # END CODING HERE ##
            ####################

            real_probs = torch.nn.functional.sigmoid(real_logits)
            fake_probs = torch.nn.functional.sigmoid(fake_logits)
            critic_accuracy += (real_probs > 0.5).sum().item() + (fake_probs < 0.5).sum().item()
            dloss_e = Dloss.item()

            ######################
            # START CODING HERE ##
            ######################
            # Step 1 - Forward pass for training the generator
            fake_logits, _ = None

            # Step 2 - Compute the loss of the generator
            # The generator wants his generated images to be positive
            Gloss = None

            # Step 3 - Reinitialize the gradient accumulator of the critic
            None

            # Step 4 - Perform the backward pass on the loss
            None

            # Step 5 - Update the parameters of the generator
            None
            ####################
            # END CODING HERE ##
            ####################

            gloss_e = Gloss.item()

            Nc += 2*bi
            tot_closs += 2 * bi * dloss_e
            Ng += bi
            tot_gloss += bi * gloss_e

        critic_accuracy /= Nc
        tot_closs /= Nc
        tot_gloss /= Ng
        logger.info(f"[Epoch {e+1}] C loss : {tot_closs} ; C accuracy : {critic_accuracy}, G loss : {tot_gloss}")

        tensorboard_writer.add_scalar("Critic loss", tot_closs, e+1)
        tensorboard_writer.add_scalar("Critic accuracy", critic_accuracy, e+1)
        tensorboard_writer.add_scalar("Generator loss", tot_gloss, e+1)

        # Generate few samples from the generator
        model.eval()
        fake_images = model.generator(X=fixed_noise)
        # Unscale the images
        fake_images = fake_images * data._MNIST_STD + data._MNIST_MEAN
        grid = torchvision.utils.make_grid(fake_images,
                                           nrow=sample_nrows,
                                           normalize=True)
        tensorboard_writer.add_image("Generated", grid, e+1)
        torchvision.utils.save_image(grid, f'images/images-{e+1:04d}.png')

        real_images = X[:sample_nrows*sample_ncols,...]
        X = X * data._MNIST_STD + data._MNIST_MEAN
        grid = torchvision.utils.make_grid(real_images,
                                           nrow=sample_nrows,
                                           normalize=True)
        tensorboard_writer.add_image("Real", grid, e+1)

        # We save the generator
        logger.info(f"Generator saved at {save_path}")
        torch.save(model.generator, save_path)


def generate(args):
    """
    Function to generate new samples from the generator
    using a pretrained network
    """

    # Parameters
    modelpath = args.modelpath
    assert(modelpath is not None)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    ######################
    # START CODING HERE ##
    ######################
    # Step 1 - Reload the generator
    generator = None

    # Put the model in evaluation mode (due to BN and Dropout)
    generator.eval()

    # Generate some samples
    sample_nrows = 1
    sample_ncols = 8

    # Step 2 - Generate a noise vector, normaly distributed
    #          of shape (sample_nrows * sample_ncol, generator.latent_size)
    z = None

    # Step 3 - Forward pass through the generator
    #          The output is (B, 1, 28, 28)
    fake_images = None

    # Denormalize the result
    fake_images = fake_images * data._MNIST_STD + data._MNIST_MEAN
    ####################
    # END CODING HERE ##
    ####################

    grid = torchvision.utils.make_grid(fake_images,
                                       nrow=sample_ncols,
                                       normalize=True)
    torchvision.utils.save_image(grid, f'generated1.png')



if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=['train', 'generate'])

    # Data parameters
    parser.add_argument("--dataset",
                        choices=["MNIST", "FashionMNIST", "EMNIST", "SVHN"],
                        help="Which dataset to use")
    parser.add_argument("--dataset_root",
                        type=str,
                        help="The root dir where the datasets are stored",
                        default=data._DEFAULT_DATASET_ROOT)
    parser.add_argument("--nthreads",
                        type=int,
                        help="The number of threads to use "
                        "for loading the data",
                        default=6)

    # Training parameters
    parser.add_argument("--num_epochs",
                        type=int,
                        help="The number of epochs to train for",
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        help="The size of a minibatch",
                        default=64)
    parser.add_argument("--base_lr",
                        type=float,
                        help="The initial learning rate to use",
                        default=0.00005)
    parser.add_argument("--debug",
                        action="store_true",
                        help="Whether to use small datasets")

    # Architectures
    parser.add_argument("--discriminator_base_c",
                        type=int,
                        help="The base number of channels for the discriminator",
                        default=32)
    parser.add_argument("--generator_base_c",
                        type=int,
                        help="The base number of channels for the generator",
                        default=64)
    parser.add_argument("--latent_size",
                        type=int,
                        help="The dimension of the latent space",
                        default=100)

    # Regularization
    parser.add_argument("--dropout",
                        type=float,
                        help="The probability of zeroing before the FC layers",
                        default=0.3)

    # For the generation
    parser.add_argument("--modelpath",
                        type=str,
                        help="The path to the pt file of the generator to load",
                        default=None)
    
    args = parser.parse_args()

    eval(f"{args.command}(args)")
