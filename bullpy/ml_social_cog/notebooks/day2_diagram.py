"""
day 2 visualize VAE architecture 
understand encoder/decoder flow
kind of didnt need to do this, just a wack diagram that i already knew, essentially just practiced matplotlib
"""

import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_vae_diagram(save_path="outputs/day2_vae_diagram.png"):
    """
    draw a clear diagram of VAE encoder/decoder flow
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # colorus 
    input_color = '#E8F4F8'
    encoder_color = '#FFE5B4'
    latent_color = '#F5C6D6'
    decoder_color = '#D8E4BC'
    output_color = '#F5C6D6'

    # input (observed signal)
    input_box = FancyBboxPatch((0.5, 2.5), 1.5, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor=input_color,
                                    edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 3, 'observed\nSignal\nx', ha='center', va='center', fontsize=10, weight='bold')

    # encoder 
    encoder_box = FancyBboxPatch((2.5, 1.5), 3, 3,
                                boxstyle="round,pad=-0.1",
                                facecolor=encoder_color,
                                edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(3.5, 3, 'Encoder', ha='center', va='center', fontsize=9, weight='bold')

    # Latent space (μ, σ)
    mu_box = FancyBboxPatch((5.2, 3.5), 0.8, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=latent_color,
                            edgecolor='black', linewidth=2)
    ax.add_patch(mu_box)
    ax.text(5.6, 3.5, 'μ', ha='center', va='center', fontsize=12, weight='bold')

    sigma_box = FancyBboxPatch((5.2, 2.2), 0.8, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor=latent_color,
                                edgecolor='black', linewidth=2)
    ax.add_patch(sigma_box)
    ax.text(5.6, 2.2, 'σ', ha='center', va='center', fontsize=12, weight='bold')

    # sampling (reparameterization trick)
    sample_circle = plt.Circle((6.5, 3), 0.4, color='red', fill=False, linewidth=2)
    ax.add_patch(sample_circle)
    ax.text(6.5, 3, 'z', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(6.5, 2.3, 'z = μ + σ·ε\nε ~ N(0,1)', ha='center', va='center', fontsize=8, style='italic')

    # decoder 
    decoder_box = FancyBboxPatch((7.5, 1.5), 2, 3,
                                boxstyle="round,pad=0.1",
                                facecolor=decoder_color,
                                edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(8.5, 3, 'Decoder\np(x|z)\n\nInput: z\nOutput: x̂',
            ha='center', va='center', fontsize=9, weight='bold')
        
    # output (reconstruction)
    output_box = FancyBboxPatch((10, 2.5), 1.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=output_color, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(10.75, 3, 'Reconstruction\nx̂', ha='center', va='center', fontsize=10, weight='bold')

    # arrows 
    # input -> encoder 
    arrow1 = FancyArrowPatch((1.25, 3), (2.5, 1.5),
                            color='black', linewidth=2,
                            arrowstyle='->', mutation_scale=20)
    ax.add_patch(arrow1)

    # Encoder -> μ, σ
    arrow2a = FancyArrowPatch((4.5, 3.8), (5.2, 3.8),
                            color='black', linewidth=2,
                            arrowstyle='->')
    ax.add_patch(arrow2a)
    arrow2b = FancyArrowPatch((4.5, 2.5), (5.2, 2.5),
                            color='black', linewidth=2,
                            arrowstyle='->')
    ax.add_patch(arrow2b)

    # μ, σ -> Sample z
    arrow3a = FancyArrowPatch((6, 3.8), (6.1, 3.2), 
                                arrowstyle='->', lw=2, color='red')
    ax.add_patch(arrow3a)

    arrow3b = FancyArrowPatch((6, 2.5), (6.1, 2.8), 
                                arrowstyle='->', lw=2, color='red')
    ax.add_patch(arrow3b)

    # z -> decoder 
    arrow4 = FancyArrowPatch((6.9, 3), (7.5, 3), 
                                arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow4)

    # decoder -> output
    arrow5 = FancyArrowPatch((9.5, 3), (10, 3), 
                                arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow5)

    # Loss components
    ax.text(1.25, 1.5, 'Loss = Reconstruction + KL Divergence', 
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(1.25, 1.1, 'Reconstruction: ||x - x̂||²\nKL: D_KL(q(z|x) || p(z))', 
            ha='center', va='center', fontsize=9, style='italic')

    plt.title("Variational Autoencoder Architecture", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved VAE diagram to {save_path}")

if __name__ == "__main__":
    draw_vae_diagram()