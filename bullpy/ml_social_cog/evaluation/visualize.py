"""
visualisations for day 1 experiment
"""

import matplotlib.pyplot as plt 
import json
import torch

def plot_loss_curve(losses, save_path="outputs/day1_loss_curve.png"):
    """
    plot training loss over epochs
    """
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("training loss curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"save loss curve to {save_path}")

def plot_predictions_vs_truth(predictions, targets, save_path="outputs/day1_predictions.png"):
    """
    plot predictions vs truth
    """
    plt.figure(figsize=(8,6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)],
             'r--', label="Perfect Prediction")
    plt.xlabel("True Value")
    plt.ylabel("predicted value")
    plt.title("predictions vs truth")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"save predictions vs truth to {save_path}")

def main():
    """
    load json files and create visualisations
    """
    #load losses 
    with open("outputs/day1_losses.json", "r") as f:
        losses_data = json.load(f)
    losses = losses_data["losses"]

    #load predictions
    with open("outputs/day1_predictions.json", "r") as f:
        predictions_data = json.load(f)
    predictions = predictions_data["predictions"]
    targets = predictions_data["targets"]

    # create plots
    plot_loss_curve(losses)
    plot_predictions_vs_truth(predictions, targets)

    print("all visualisations saved!")

if __name__ == "__main__":
    main()