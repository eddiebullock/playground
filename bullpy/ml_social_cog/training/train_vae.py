"""
vae training loop 
train vaw with proper loss (reconstruction + kl divergence)
"""

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from models.vae import VAE, vae_loss

def train_vae_epoch(model: VAE, dataloader: DataLoader, 
                    optimizer: optim.Optimizer, beta: float = 1.0) -> dict:
    """
    train vae for one epoch

    returns:
        dictionary with total loss, recon loss kl loss
    """
    model.train()
    total_recon = 0.0 
    total_kl = 0.0
    total_loss = 0.0
    n_batches = 0

    for x_batch, _ in dataloader: # note vae doesnt need labels
        # zero gradients
        optimizer.zero_grad()

        # forward pass
        x_recon, mu, logvar = model(x_batch)
        # compute loss
        loss, recon_loss, kl_loss = vae_loss(x_batch, x_recon, mu, logvar, beta=beta)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # accumulate losses 
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches
    }

def train_vae(model: VAE, train_loader: DataLoader,
            n_epochs: int = 50, learning_rate: float = 0.001,
            beta: float = 1.0) -> dict:
            """ 
            full training loop 
            returns:
                dict with loss histories 
            """
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            losses = {'total': [], 'recon': [], 'kl': []}

            print("training vae...")
            for epoch in range(n_epochs):
                epoch_losses = train_vae_epoch(model, train_loader, optimizer, beta=beta)

                losses['total'].append(epoch_losses['total_loss'])
                losses['recon'].append(epoch_losses['recon_loss'])
                losses['kl'].append(epoch_losses['kl_loss'])

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"epoch {epoch+1}/{n_epochs}:")
                    print(f"total loss: {losses['total'][-1]:.4f}")
                    print(f"recon loss: {losses['recon'][-1]:.4f}")
                    print(f"kl loss: {losses['kl'][-1]:.4f}")

            return losses