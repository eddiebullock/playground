"""
vae visualisation utilities 
purpose visualise latent space and reconstruction and training progress
"""

import torch 
import matplotlib.pyplot as plt
import numpy as np 
from models.vae import VAE
from torch.utils.data import DataLoader

def plot_latent_space(model: VAE, dataloader: DataLoader, save_path="outputs/latent_space.png"):
    """
    Visualize latent space: plot μ for each data point.
    If latent_dim=2, we can plot in 2D.
    """
    model.eval()
    latents = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            # encode to latent space 
            z = model.encode(x_batch) # batch_size, latent_dim
            latents.append(z.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    latents = np.concatenate(latents, axis=0) # (n_samples, latent_dim)
    targets = np.concatenate(targets, axis=0) # (n_samples,)

    if latents.shape[1] == 2:
        # 2d latent space 
        # plt.figure(figsize-(10, 8))
        scatter = plt.scatter(latents[:, 0], latents[:, 1],
                            c=targets, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='True hidden state')
        plt.xlabel('latent dimension')
        plt.ylabel('latent dimension')
        plt.title('latent space vis')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved latent space plot to {save_path}")
    else:
        print(f"latent space is {latents.shape[1]}, skippin plotting")

def plot_reconstruction(model: VAE, dataloader: DataLoader, n_samples=8, 
                        save_path="outputs/reconstruction.png"):
    """
    plot original vs reconstructed samples 
    """
    model.eval()

    # get one batch 
    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch[:n_samples] # take first n samples

    with torch.no_grad():
        x_recon, _, _ = model(x_batch)

    x_batch = x_batch.cpu().numpy()
    x_recon = x_recon.cpu().numpy()

    # plot 
    n_subplots = n_samples  # Define the number of subplots
    fig, axes = plt.subplots(2, n_subplots, figsize=(2 * n_samples, 4))
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_samples):
        # original
        axes[0, i].plot(x_batch[i], 'b-', label='original')
        axes[0, i].set_title(f'Sample {i + 1}: original')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim([x_batch.min(), x_batch.max()])

        # reconstruction 
        axes[1, i].plot(x_recon[i], 'r--', label='reconstruction')
        axes[1, i].set_title(f'Sample {i + 1}: reconstructed')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim([x_batch.min(), x_batch.max()])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved reconstruction plot to {save_path}")

def plot_vae_loss_curve(losses, save_path='outputs/vae_loss_curve.png'):
    """
    plot VAE loss components total, reconstruction, kl divergence
    """
    plt.figure(figsize=(10, 6))

    plt.plot(losses['total'], label='Total loss', linewidth=2)
    plt.plot(losses['recon'], label='reconstruction loss', linewidth=2)
    plt.plot(losses['kl'], label='kl divergence loss', linewidth=2, linestyle=':')

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('VAE loss curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"save loss curve to {save_path}")
    
    