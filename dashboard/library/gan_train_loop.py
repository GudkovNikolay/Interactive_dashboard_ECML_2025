import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from timeit import default_timer as timer
from IPython.display import clear_output

from library.constants import DEVICE, N_ASSETS, WINDOW_SIZE
from library.correlations import plot_correlation_matrix
from library.generation import generate_samples

SAVE_PATH = Path('models/')
SAVE_PATH.mkdir(exist_ok=True)

loss_fn = nn.BCELoss()


def train_epoch(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader) -> tuple[float, float]:
    """
    Train GAN
    Return Generator and Discriminator losses
    """
    generator.train()
    discriminator.train()

    generator_losses = []
    discriminator_losses = []
    # Iterate over batches of real samples
    for idx, real_samples in enumerate(dataloader):
        real_samples = real_samples.to(DEVICE)

        # Generate fake samples from the generator
        # The same noise will be used in Generator and Discriminator training
        z = generator.get_noise(real_samples.shape[0]).to(DEVICE)
        with torch.no_grad():
            fake_samples = generator(z)
        real_labels = torch.ones(real_samples.shape[0]).to(DEVICE)
        fake_labels = torch.zeros(real_samples.shape[0]).to(DEVICE)

        # Train the discriminator
        discriminator_optimizer.zero_grad()
        # Compute discriminator loss on real samples
        real_loss = loss_fn(discriminator(real_samples), real_labels)
        # Compute discriminator loss on fake samples
        fake_loss = loss_fn(discriminator(fake_samples), fake_labels)
        # Compute the total discriminator loss
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Clip weights
        clip = 0.01
        for dp in discriminator.parameters():
            dp.data.clamp_(-clip, clip)

        if idx % 2 == 0:
            # Train the generator
            generator_optimizer.zero_grad()
            # Generate fake samples and compute generator loss
            fake_samples = generator(z)
            generator_loss = loss_fn(discriminator(fake_samples), real_labels)
            generator_loss.backward()
            generator_optimizer.step()

        discriminator_losses.append(discriminator_loss.item())
        generator_losses.append(generator_loss.item())
    return np.mean(generator_losses), np.mean(discriminator_losses)


@torch.no_grad()
def plot_gan(generator, assets: list[str], generator_losses: list[float], discriminator_losses: list[float], epoch: int, df_returns_real: pd.DataFrame):
    """
    Print statistics
    Plot distribution
    Plot correlation matrices
    Plot cumulative returns
    """

    plot_columns = ['SBER', 'SBERP']

    # Generate fake DataFrame
    df_returns_fake = generate_samples(generator, assets)

    # Print statistics
    print(f'Fake std: {df_returns_fake.std(axis=0).values}.\nReal std: {df_returns_real.std(axis=0).values}')
    print(f'Fake correlation: {df_returns_fake[plot_columns].corr().iloc[0][1]}. Real correlation: {df_returns_real[plot_columns].corr().iloc[0][1]}')

    # Plot returns distribution
    plt.subplots(1, 2, figsize=(15, 5))
    for i, col in enumerate(plot_columns):
        plt.subplot(1, 2, i + 1)

        # Plot returns distributions
        sns.histplot(df_returns_real[col], stat='density', label='real')
        sns.histplot(df_returns_fake[col], stat='density', label='fake')

        # Plot real returns bounds
        plt.axvline(df_returns_real[col].min(), linestyle='dashed', color='C0')
        plt.axvline(df_returns_real[col].max(), linestyle='dashed', color='C0')

        # Plot fake returns bounds
        plt.axvline(df_returns_fake[col].min(), linestyle='dashed', color='C1')
        plt.axvline(df_returns_fake[col].max(), linestyle='dashed', color='C1')

        plt.legend(loc='upper left')
        plt.title(f'{col} ({epoch} epoch)')
    plt.show()

    # Plot correlation matrices
    plt.subplots(1, 2, figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title('Real')
    sorted_labels = plot_correlation_matrix(df_returns_real.corr())

    plt.subplot(1, 2, 2)
    plt.title('Fake')
    plot_correlation_matrix(df_returns_fake.corr(), sorted_labels)

    plt.show()

    # Plot cumulative returns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    plt.title('Real')
    plt.ylabel('Cumulative return')
    df_returns_real.iloc[:WINDOW_SIZE].cumsum().plot(ax=ax1)

    plt.subplot(1, 2, 2)
    plt.title('Fake')
    plt.ylabel('Cumulative return')
    df_returns_fake.set_index(df_returns_real.index[:WINDOW_SIZE]).cumsum().plot(ax=ax2)

    plt.show()

    # Plot losses
    plt.subplots(1, 2, figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), generator_losses)
    plt.title('Generator Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), discriminator_losses)
    plt.title('Discriminator Loss')

    plt.show()


def save_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch: int, model_prefix: str):
    """
    Save GAN checkpoint
    """
    model_path = SAVE_PATH / model_prefix
    model_path.mkdir(exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
    }, model_path / f'checkpoint_{epoch}')


def load_gan(model_prefix: str, generator=None, discriminator=None, generator_optimizer=None, discriminator_optimizer=None, epoch: int | None = None):
    """
    Load GAN checkpoint
    Load only models that are not None
    Load latest epoch if not specified
    """
    model_path = SAVE_PATH / model_prefix
    assert model_path.exists()
    if epoch is None:
        # Find latest checkpoint
        files = list(model_path.iterdir())
        assert len(files) > 0
        for file in files:
            assert file.name.startswith('checkpoint_')
        epochs = [int(file.name.removeprefix('checkpoint_')) for file in files]
        epoch = max(epochs)

    print(f'Load {epoch} epoch checkpoint')
    checkpoint = torch.load(model_path / f'checkpoint_{epoch}')
    assert checkpoint['epoch'] == epoch

    # Load models
    if generator is not None:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator.eval()

    # Load optimizers
    if generator_optimizer is not None:
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    if discriminator_optimizer is not None:
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])


def train_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader, df_returns_real: pd.DataFrame, n_epochs: int, log_frequency: int, save_frequency: int, model_prefix: str) -> tuple[list[float], list[float]]:
    """
    Train gan
    """
    torch.manual_seed(1)
    start = timer()

    # Save losses on each epoch
    generator_losses = []
    discriminator_losses = []

    for epoch in range(1, n_epochs + 1):
        # Train one epoch
        generator_loss, discriminator_loss = train_epoch(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader)

        # Store losses
        generator_losses.append(generator_loss)
        discriminator_losses.append(discriminator_loss)

        # Plot samples
        if epoch % log_frequency == 0 or epoch == n_epochs:
            # Clear output
            clear_output(wait=True)
            # Log time
            train_time = timer() - start
            print(f'{log_frequency} epochs train time: {train_time:.1f}s. Estimated train time: {((n_epochs - epoch) * train_time / log_frequency / 60):.1f}m')
            start = timer()
            # Plot samples
            plot_gan(generator, dataloader.dataset.assets, generator_losses, discriminator_losses,  epoch, df_returns_real)

        # Save model
        if epoch % save_frequency == 0 or epoch == n_epochs:
            save_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch, model_prefix)

    # Return losses
    return generator_losses, discriminator_losses
