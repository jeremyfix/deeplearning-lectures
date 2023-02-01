#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import argparse
import logging
import sys
import os
from typing import Callable, Dict

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
    wdecay = args.wdecay
    num_epochs = args.num_epochs
    discriminator_base_c = args.discriminator_base_c
    generator_base_c = args.generator_base_c
    latent_size = args.latent_size
    sample_nrows = 8
    sample_ncols = 8

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Dataloaders
    train_loader, valid_loader, img_shape = data.get_dataloaders(
        dataset_root=dataset_root,
        cuda=use_cuda,
        batch_size=batch_size,
        n_threads=nthreads,
        dataset=dataset,
        small_experiment=debug,
    )

    # Model definition
    model = models.GAN(
        img_shape, dropout, discriminator_base_c, latent_size, generator_base_c
    )
    model.to(device)

    # Optimizers
    critic = model.discriminator
    generator = model.generator
    ######################
    # START CODING HERE ##
    ######################
    # Step 1 - Define the optimizer for the critic
    # @TEMPL@optim_critic = None
    # @SOL
    if wdecay == 0:
        print("No weight decay")
        optim_critic = optim.Adam(critic.parameters(), lr=base_lr)
    else:
        optim_critic = optim.AdamW(critic.parameters(), lr=base_lr, weight_decay=wdecay)

    # SOL@
    # Step 2 - Define the optimizer for the generator
    # @TEMPL@optim_generator = None
    # @SOL
    optim_generator = optim.Adam(generator.parameters(), lr=base_lr)
    # SOL@

    # Step 3 - Define the loss (it must embed the sigmoid)
    # @TEMPL@loss = None
    loss = torch.nn.BCEWithLogitsLoss()  # @SOL@

    ####################
    # END CODING HERE ##
    ####################

    # Callbacks
    summary_text = (
        "## Summary of the model architecture\n"
        + f"{deepcs.display.torch_summarize(model)}\n"
    )
    summary_text += "\n\n## Executed command :\n" + "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    if args.logdir is None:
        logdir = generate_unique_logpath("./logs", "gan")
    else:
        logdir = args.logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    tensorboard_writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )

    with open(os.path.join(logdir, "summary.txt"), "w") as f:
        f.write(summary_text)

    save_path = os.path.join(logdir, "generator.pt")

    logger.info(f">>>>> Results saved in {logdir}")

    # Note: the validation data are only real images, hence the metrics below
    val_fmetrics = {
        "accuracy": lambda real_probas: (real_probas > 0.5).double().mean(),
        "loss": lambda real_probas: -real_probas.log().mean(),
    }

    # Define a fixed noise used for sampling
    fixed_noise = torch.randn(sample_nrows * sample_ncols, latent_size).to(device)

    # Generate few samples from the initial generator
    model.eval()
    fake_images = model.generator(X=fixed_noise)
    grid = torchvision.utils.make_grid(fake_images, nrow=sample_nrows, normalize=True)
    tensorboard_writer.add_image("Generated", grid, 0)

    imgpath = logdir + "/images/"
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)
    torchvision.utils.save_image(grid, imgpath + "images-0000.png")

    # Training loop
    for e in range(num_epochs):

        tot_dploss = tot_dnloss = tot_gloss = 0
        critic_paccuracy = critic_naccuracy = 0
        Ns = 0

        model.train()
        for ei, (X, _) in enumerate(tqdm.tqdm(train_loader)):

            # X is a batch of real data
            X = X.to(device)
            bi = X.shape[0]

            pos_labels = torch.ones((bi,)).to(device)
            neg_labels = torch.zeros((bi,)).to(device)

            ######################
            # START CODING HERE ##
            ######################
            # Step 1 - Forward pass for training the discriminator
            # @TEMPL@real_logits, _ = None
            # @TEMPL@fake_logits, _ = None
            real_logits, _ = model(X, None)  # @SOL@
            fake_logits, _ = model(None, bi)  # @SOL@

            # Step 2 - Compute the loss of the critic
            # @TEMPL@Dloss = None + None
            # @SOL
            D_ploss = loss(real_logits, pos_labels)
            D_nloss = loss(fake_logits, neg_labels)
            Dloss = 0.5 * (D_ploss + D_nloss)
            # SOL@

            # Step 3 - Reinitialize the gradient accumulator of the critic
            # @TEMPL@None
            optim_critic.zero_grad()  # @SOL@

            # Step 4 - Perform the backward pass on the loss
            # @TEMPL@None
            Dloss.backward()  # @SOL@

            # Step 5 - Update the parameters of the critic
            # @TEMPL@None
            optim_critic.step()  # @SOL@

            ####################
            # END CODING HERE ##
            ####################

            real_probs = torch.sigmoid(real_logits)
            fake_probs = torch.sigmoid(fake_logits)
            critic_paccuracy += (real_probs > 0.5).sum().item()
            critic_naccuracy += (fake_probs < 0.5).sum().item()
            dploss_e = Dloss.item()
            dnloss_e = Dloss.item()

            ######################
            # START CODING HERE ##
            ######################
            # Step 1 - Forward pass for training the generator
            # @TEMPL@fake_logits, _ = None
            fake_logits, _ = model(None, bi)  # @SOL@

            # Step 2 - Compute the loss of the generator
            # The generator wants his generated images to be positive
            # @TEMPL@Gloss = None
            Gloss = loss(fake_logits, pos_labels)  # @SOL@

            # Step 3 - Reinitialize the gradient accumulator of the critic
            # @TEMPL@None
            optim_generator.zero_grad()  # @SOL@

            # Step 4 - Perform the backward pass on the loss
            # @TEMPL@None
            Gloss.backward()  # @SOL@

            # Step 5 - Update the parameters of the generator
            # @TEMPL@None
            optim_generator.step()  # @SOL@
            ####################
            # END CODING HERE ##
            ####################

            gloss_e = Gloss.item()

            tot_dploss += bi * dploss_e
            tot_dnloss += bi * dnloss_e
            tot_gloss += bi * gloss_e
            Ns += bi

        critic_paccuracy /= Ns
        critic_naccuracy /= Ns
        tot_dploss /= Ns
        tot_dnloss /= Ns
        tot_gloss /= Ns

        # Evaluate the metrics on the validation set
        val_metrics = evaluate(model, device, valid_loader, val_fmetrics)

        logger.info(
            f"[Epoch {e+1}] "
            f"D ploss : {tot_dploss:.4f} ; "
            f"D paccuracy : {critic_paccuracy:.2f}, "
            f"D nloss : {tot_dnloss:.4f} ; "
            f"D naccuracy : {critic_naccuracy:.2f}, "
            f"D vloss :  {val_metrics['loss']:.4f}; "
            f"D vaccuracy : {val_metrics['accuracy']:.2f}, "
            f"G loss : {tot_gloss:.4f}"
        )

        tensorboard_writer.add_scalar("Critic p-loss", tot_dploss, e + 1)
        tensorboard_writer.add_scalar("Critic n-loss", tot_dnloss, e + 1)
        tensorboard_writer.add_scalar("Critic v-loss", val_metrics["loss"], e + 1)
        tensorboard_writer.add_scalar("Critic p-accuracy", critic_paccuracy, e + 1)
        tensorboard_writer.add_scalar("Critic n-accuracy", critic_naccuracy, e + 1)
        tensorboard_writer.add_scalar(
            "Critic v-accuracy", val_metrics["accuracy"], e + 1
        )
        tensorboard_writer.add_scalar("Generator loss", tot_gloss, e + 1)

        # Generate few samples from the generator
        model.eval()
        fake_images = model.generator(X=fixed_noise)
        # Unscale the images
        fake_images = fake_images * data._IMG_STD + data._IMG_MEAN
        grid = torchvision.utils.make_grid(
            fake_images, nrow=sample_nrows, normalize=True
        )
        tensorboard_writer.add_image("Generated", grid, e + 1)
        torchvision.utils.save_image(grid, imgpath + f"images-{e+1:04d}.png")

        X, _ = next(iter(train_loader))
        real_images = X[: (sample_nrows * sample_ncols), ...]
        real_images = real_images * data._IMG_STD + data._IMG_MEAN
        grid = torchvision.utils.make_grid(
            real_images, nrow=sample_nrows, normalize=True
        )
        tensorboard_writer.add_image("Real", grid, e + 1)
        # torchvision.utils.save_image(grid, imgpath + f"real-{e+1:04d}.png")

        # We save the generator
        logger.info(f"Generator saved at {save_path}")
        torch.save(model.generator, save_path)

        # Important: ensure the model is in eval mode before exporting !
        # the graph in train/test mode is not the same
        model.eval()
        dummy_input = torch.zeros((1, latent_size), device=device)
        torch.onnx.export(
            model.generator,
            dummy_input,
            logdir + "generator.onnx",
            verbose=False,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )  # At least opset 11 is required otherwise it seems nn.UpSample is not correctly handled


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
    metrics: Dict[str, Callable],
):
    """
    Compute the averaged metrics given in the dictionnary.
    The dictionnary metrics gives a function to compute the metrics on a
    minibatch and averaged on it.
    """
    model.eval()

    tot_metrics = {m_name: 0.0 for m_name in metrics}
    Ns = 0
    for (inputs, targets) in loader:

        # Move the data to the GPU if required
        inputs, targets = inputs.to(device), targets.to(device)

        batch_size = inputs.shape[0]

        # Forward pass
        logits, _ = model(inputs, None)
        probas = logits.sigmoid()

        # Compute the metrics
        for m_name, m_f in metrics.items():
            tot_metrics[m_name] += batch_size * m_f(probas).item()
        Ns += batch_size

    # Size average the metrics
    for m_name, m_v in tot_metrics.items():
        tot_metrics[m_name] = m_v / Ns

    return tot_metrics


def generate(args):
    """
    Function to generate new samples from the generator
    using a pretrained network
    """

    # Parameters
    modelpath = args.modelpath
    assert modelpath is not None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    ######################
    # START CODING HERE ##
    ######################
    # Step 1 - Reload the generator
    # @TEMPL@generator = None
    generator = torch.load(modelpath).to(device)  # @SOL@

    # Put the model in evaluation mode (due to BN and Dropout)
    generator.eval()

    # Generate some samples
    sample_nrows = 1
    sample_ncols = 8

    # Step 2 - Generate a noise vector, normaly distributed
    #          of shape (sample_nrows * sample_ncol, generator.latent_size)
    # @TEMPL@z = None
    # @SOL
    z = torch.randn(sample_nrows * sample_ncols, generator.latent_size).to(device)
    # SOL@

    # Step 3 - Forward pass through the generator
    #          The output is (B, 1, 28, 28)
    # @TEMPL@fake_images = None
    fake_images = generator(z)  # @SOL@

    # Denormalize the result
    fake_images = fake_images * data._IMG_STD + data._IMG_MEAN
    ####################
    # END CODING HERE ##
    ####################

    grid = torchvision.utils.make_grid(fake_images, nrow=sample_ncols, normalize=True)
    torchvision.utils.save_image(grid, "generated1.png")

    # @SOL
    # Interpolate in the laten space
    N = 20
    z = torch.zeros((N, N, generator.latent_size)).to(device)
    # Generate the 3 corner samples
    z[0, 0, :] = torch.randn(generator.latent_size)
    z[-1, 0, :] = torch.randn(generator.latent_size)
    z[0, -1, :] = torch.randn(generator.latent_size)
    di = z[-1, 0, :] - z[0, 0, :]
    dj = z[0, -1, :] - z[0, 0, :]
    for i in range(0, N):
        for j in range(0, N):
            z[i, j, :] = z[0, 0, :] + i / (N - 1) * di + j / (N - 1) * dj
    fake_images = generator(z.reshape(N**2, -1))
    fake_images = fake_images * data._IMG_STD + data._IMG_MEAN
    grid = torchvision.utils.make_grid(fake_images, nrow=N, normalize=True)
    torchvision.utils.save_image(grid, "generated2.png")
    # SOL@


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "generate"])

    # Data parameters
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "FashionMNIST", "EMNIST", "SVHN", "CelebA"],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="The root dir where the datasets are stored",
        default=data._DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use " "for loading the data",
        default=6,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="The logdir in which to save the assets of the experiments",
        default=None,
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs", type=int, help="The number of epochs to train for", default=200
    )
    parser.add_argument(
        "--batch_size", type=int, help="The size of a minibatch", default=64
    )
    parser.add_argument(
        "--base_lr", type=float, help="The initial learning rate to use", default=0.0005
    )
    parser.add_argument(
        "--wdecay", type=float, help="The weight decay used for the critic", default=1.0
    )
    parser.add_argument(
        "--debug", action="store_true", help="Whether to use small datasets"
    )

    # Architectures
    parser.add_argument(
        "--discriminator_base_c",
        type=int,
        help="The base number of channels for the discriminator",
        default=32,
    )
    parser.add_argument(
        "--generator_base_c",
        type=int,
        help="The base number of channels for the generator",
        default=128,
    )
    parser.add_argument(
        "--latent_size", type=int, help="The dimension of the latent space", default=100
    )

    # Regularization
    parser.add_argument(
        "--dropout",
        type=float,
        help="The probability of zeroing before the FC layers",
        default=0.3,
    )

    # For the generation
    parser.add_argument(
        "--modelpath",
        type=str,
        help="The path to the pt file of the generator to load",
        default=None,
    )

    args = parser.parse_args()

    eval(f"{args.command}(args)")
