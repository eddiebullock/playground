"""
day 1 experiment: train a simple feedforward net on toy data
"""

import torch 
from torch.utils.data import DataLoader

from data.synthetic import ToyDataset
from models.feedforward import FeedforwardNet
from training.train import train_model 

def main():
    # set random seed for reproducibility
    torch.manual_seed(42)

    #create dataset
    print("creating dataset...")
    dataset = ToyDataset(n_samples=1000, input_dim=10, noise_level=0.1)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    #create model
    print("creating model...")
    model = FeedforwardNet(input_dim=10, hidden_dim=32, output_dim=1)

    #train model
    losses = train_model(model, train_loader, n_epochs=20, learning_rate=0.001)

    #check did loss decrease 
    print(f"\ninitial loss: {losses[0]:.4f}")
    print(f"final loss: {losses[-1]:.4f}")
    print(f"loss decreased: {losses[0] - losses[-1]:.4f}")

    return model, losses

if __name__ == "__main__":
    model, losses = main()