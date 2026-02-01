"""
Train CVAE (Sohn et al., 2015). One epoch: loop, forward, loss (recon + KL), backward, step.
Paper: Sohn et al., 2015 – training via SGVB; loss = reconstruction + KL (conditional ELBO).
"""

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_paper_epoch(model, loader: DataLoader, optimizer, device=None):
    """
    one epoch loop over loader, forward, placeholder loss (recon + KL), backward, step.
    returns scalar average loss for logging 
    """
    if device is None:
        device = next(model.parameters()).device 
    model.train()
    total_loss = 0.0 
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, y)

        recon = F.mse_loss(x_recon, x)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon + kl

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches 
