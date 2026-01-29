"""
training loop 
train model and watch loss decrease 
"""

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader 

def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module) -> float:
    """
    train model for one epoch 

    returns: average loss over the epoch
    """
    model.train() # set model to training mode 
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        #zero gradieints
        optimizer.zero_grad()
        # forward pass
        predictions = model(X_batch) #(batch_size, output_dim)
        #compute loss
        loss = criterion(predictions.squeeze(), y_batch) # squeez id output+dim = 1
        #backward pass
        loss.backward()
        #update weoghts (grad descent step)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches 

def train_model(model: nn.Module, train_loader: DataLoader,
                n_epochs: int = 10, learning_rate: float = 0.001):
            """
            full training loop
            """
            # loss function: mean squared error (for regression)
            criterion = nn.MSELoss()

            #optimizer: adam (adaptive learning rate)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            #training loop
            losses = []
            print("training...")
            for epoch in range(n_epochs):
                avg_loss = train_epoch(model, train_loader, optimizer, criterion)
                losses.append(avg_loss)
                print(f"epoch {epoch+1}/{n_epochs}: Loss: {avg_loss:.4f}")
            return losses