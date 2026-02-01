"""
Day 3: CVAE (Sohn et al., 2015) skeleton experiment.
Load config, build dataset/loader/model, run 1–2 epochs to confirm forward + backward.
"""

import torch 
import yaml 
from pathlib import Path 
from torch.utils.data import DataLoader

from data.synthetic import ToyDataset
from models.paper_model import ConditionalVAE
from training.paper_train import train_paper_epoch

def load_config(path="configs/day3_paper_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    torch.manual_seed(config["seed"])

    data_cfg = config["data"]
    dataset = ToyDataset(
        n_samples=data_cfg["n_samples"],
        input_dim=data_cfg["input_dim"],
        noise_level=data_cfg["noise_level"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    model = ConditionalVAE(**config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    n_epochs = config["training"]["epochs"]

    print("Day 3 skeleton: CVAE forward + backward check")
    for epoch in range (n_epochs):
        loss = train_paper_epoch(model, loader, optimizer)
        print(f"epoch {epoch + 1}/{n_epochs} loss: {loss:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()