"""
day 2 experiment: train a VAE on toy data
purpose learn latent representations of hidden states 
"""

import torch 
import json 
from torch.utils.data import DataLoader

from data.synthetic import ToyDataset
from models.vae import VAE
from training.train_vae import train_vae
from evaluation.visualize_vae import plot_latent_space, plot_reconstruction, plot_vae_loss_curve

def main():
    # set random seed for reproducibility 
    torch.manual_seed(42)

    # create dataset
    print("creating dataset...")
    dataset = ToyDataset(n_samples=1000, input_dim=10, noise_level=0.1)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # create model
    print("creating model...")
    model = VAE(input_dim=10, hidden_dim=32, latent_dim=2) # 2d latent space for visualization

    # train 
    losses = train_vae(model, train_loader, n_epochs=50, learning_rate=0.001, beta=1.0)

    # save model 
    torch.save(model.state_dict(), "outputs/day2_vae_model.pt")
    print("saved model to outputs/day2_vae_model.pt")

    # save losses
    with open("outputs/day2_losses.json", 'w') as f:
        json.dump(losses, f)
    print("saved losses to outputs/day2_losses.json")

    # visualisations
    print("\n creating visualisations...")
    plot_vae_loss_curve(losses)
    plot_latent_space(model, test_loader)
    plot_reconstruction(model, test_loader, n_samples=8)

    # print summary
    print("\n" + "="*50)
    print("training summary")
    print(f"inital total loss: {losses['total'][0]:.4f}")
    print(f"final total loss: {losses['total'][-1]:.4f}")
    print(f"Initial recon loss: {losses['recon'][0]:.4f}")
    print(f"Final recon loss: {losses['recon'][-1]:.4f}")
    print(f"initial kl loss: {losses['kl'][0]:.4f}")
    print(f"final kl loss: {losses['kl'][-1]:.4f}")
    print("="*50 + "\n")

    return model, losses

if __name__ == "__main__":
    model, losses = main()
