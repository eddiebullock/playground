"""
Day 3: CVAE architecture diagram (Sohn et al., 2015)
Learning Structured Output Representation using Deep Conditional Generative Models

Diagram shows: inputs (x, y), ENCODER q(z|x,y), latent (mu, sigma), sample z,
DECODER p(x|z,y), reconstruction x_hat, and LOSS.
Annotations in code mark encoder, decoder, and loss for reference.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
from pathlib import Path

# Project root (parent of notebooks/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def draw_cvae_diagram(save_path=None):
    """
    Draw CVAE (Conditional VAE) architecture from Sohn et al. 2015.
    - This box is the ENCODER: q(z|x,y), maps (x,y) -> mu, logvar.
    - This box is the DECODER: p(x|z,y), maps (z,y) -> x_recon.
    - Loss is shown at bottom: conditional ELBO = reconstruction + KL.
    """
    if save_path is None:
        save_path = PROJECT_ROOT / "outputs" / "day3_architecture.png"
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Colors (match day2 style)
    input_color = "#E8F4F8"
    condition_color = "#E8E8F8"
    encoder_color = "#FFE5B4"   # ENCODER
    latent_color = "#F5C6D6"
    decoder_color = "#D8E4BC"   # DECODER
    output_color = "#F5C6D6"
    loss_color = "#FFFF99"

    # ---- INPUTS: x (observed) and y (condition) ----
    input_x = FancyBboxPatch((0.3, 2.4), 1.2, 1.0,
                             boxstyle="round,pad=0.1",
                             facecolor=input_color, edgecolor="black", linewidth=2)
    ax.add_patch(input_x)
    ax.text(0.9, 2.9, "x\n(observed)", ha="center", va="center", fontsize=9, weight="bold")

    input_y = FancyBboxPatch((0.3, 3.8), 1.2, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor=condition_color, edgecolor="black", linewidth=2)
    ax.add_patch(input_y)
    ax.text(0.9, 4.25, "y\n(condition)", ha="center", va="center", fontsize=9, weight="bold")

    # ---- ENCODER: q(z|x,y) ---- (this box is the encoder)
    encoder_box = FancyBboxPatch((2.0, 1.4), 3.2, 3.2,
                                boxstyle="round,pad=0.1",
                                facecolor=encoder_color, edgecolor="black", linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(3.6, 3.0, "Encoder\nq(z|x,y)", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(3.6, 2.3, "Input: x, y\nOutput: mu, logvar", ha="center", va="center", fontsize=8)

    # Latent parameters mu, sigma
    mu_box = FancyBboxPatch((5.4, 3.4), 0.7, 0.7,
                            boxstyle="round,pad=0.05",
                            facecolor=latent_color, edgecolor="black", linewidth=2)
    ax.add_patch(mu_box)
    ax.text(5.75, 3.75, "mu", ha="center", va="center", fontsize=10, weight="bold")

    sigma_box = FancyBboxPatch((5.4, 2.2), 0.7, 0.7,
                               boxstyle="round,pad=0.05",
                               facecolor=latent_color, edgecolor="black", linewidth=2)
    ax.add_patch(sigma_box)
    ax.text(5.75, 2.55, "sigma", ha="center", va="center", fontsize=9, weight="bold")

    # Sample z (reparameterization)
    sample_circle = plt.Circle((6.8, 2.95), 0.45, color="red", fill=False, linewidth=2)
    ax.add_patch(sample_circle)
    ax.text(6.8, 2.95, "z", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(6.8, 2.0, "z = mu + sigma*eps", ha="center", va="center", fontsize=7, style="italic")

    # ---- DECODER: p(x|z,y) ---- (this box is the decoder)
    decoder_box = FancyBboxPatch((8.0, 1.4), 2.5, 3.2,
                                boxstyle="round,pad=0.1",
                                facecolor=decoder_color, edgecolor="black", linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(9.25, 3.0, "Decoder\np(x|z,y)", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(9.25, 2.3, "Input: z, y\nOutput: x_hat", ha="center", va="center", fontsize=8)

    # Output reconstruction
    output_box = FancyBboxPatch((11.0, 2.5), 1.2, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor=output_color, edgecolor="black", linewidth=2)
    ax.add_patch(output_box)
    ax.text(11.6, 3.0, "x_hat\n(recon)", ha="center", va="center", fontsize=9, weight="bold")

    # ---- ARROWS ----
    # x, y -> encoder
    ax.annotate("", xy=(2.0, 2.6), xytext=(1.5, 2.9),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(2.0, 3.2), xytext=(1.5, 4.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    # encoder -> mu, sigma
    ax.annotate("", xy=(5.4, 3.75), xytext=(5.2, 3.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(5.4, 2.55), xytext=(5.2, 2.6),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    # mu, sigma -> z
    ax.annotate("", xy=(6.35, 2.95), xytext=(6.1, 3.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"))
    ax.annotate("", xy=(6.35, 2.95), xytext=(6.1, 2.55),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"))
    # z -> decoder (and y into decoder)
    ax.annotate("", xy=(8.0, 2.95), xytext=(7.25, 2.95),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(8.5, 3.8), xytext=(7.0, 4.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="gray", linestyle="--"))
    ax.text(7.5, 4.2, "y", fontsize=8, color="gray")
    # decoder -> output
    ax.annotate("", xy=(11.0, 3.0), xytext=(10.5, 3.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # ---- LOSS (annotated: loss is here) ----
    loss_box = FancyBboxPatch((2.5, 0.25), 7.0, 0.85,
                               boxstyle="round,pad=0.08",
                               facecolor=loss_color, edgecolor="black", linewidth=2)
    ax.add_patch(loss_box)
    ax.text(6.0, 0.9, "Loss: Conditional ELBO", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(6.0, 0.5, "Reconstruction: ||x - x_hat||^2  +  KL(q(z|x,y) || p(z|y))", ha="center", va="center", fontsize=9)

    ax.set_title("CVAE Architecture (Sohn et al., 2015)\nConditional VAE for Structured Output Prediction", fontsize=12, weight="bold")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved CVAE diagram to {save_path}")


if __name__ == "__main__":
    draw_cvae_diagram()
