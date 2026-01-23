"""
synthetic data generator for feedforward NN
purpose: create simple hidden-state -> noisy signal data for day1
"""

import torch 
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    """ 
    simple synth dataset: hidden state -> noisy observation 

    True process
        hidden state (1D) -> signal (higher dim) + noise

    This mimis: we observe noisy signal and want to infer hidden state.
    """

    def __init__(self, n_samples: int = 1000, input_dim: int = 10, noise_level: float = 0.1):
        """
        args:
            n_samples: number of datapoints
            input_dim: dimension of observed signal 
            noise_level: standard deviation of noise
        """
        self.n_samples = n_samples
        self.input_dim = input_dim 
        self.noise_level = noise_level

        # generate data 
        # tru hidden state (simple for now just scalar that determines signal)
        hidden_states = torch.randn(n_samples, 1) # (n_samples, 1)

        # generate signal: hidden_state projects to higher dim space 
        # this is the true mapping we want to learn 
        projection = torch.randn(1, input_dim) # random projection matrix 
        signals = hidden_states @ projection # (n_samples, input_dim)

        # add noise to signal 
        noise = torch.randn(n_samples, input_dim) * noise_level
        self.X = signals + noise # observed signal 

        # target: predict hidden states (regression task)
        self.y = hidden_states.squeeze() # (n_samples,)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#test the dataset
if __name__ == "__main__":
    dataset = ToyDataset(n_samples=100, input_dim=10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    #check one batch 
    X_batch, y_batch = next(iter(dataloader))
    print(f"X_batch shape: {X_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}") 
