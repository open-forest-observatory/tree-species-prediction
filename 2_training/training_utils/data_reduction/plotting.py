import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path

def plot_projection_sorted(proj_vals, save_dir, epoch=None, title_prefix="OMP", figsize=(12,8), plot_style='notebook'):
    """
    Plots sorted |Aᵀb| magnitudes in descending order.
    Shows how sharply OMP's projection spectrum decays.
    """
    sns.set_context(plot_style)
    vals = np.asarray(proj_vals, dtype=float)
    sorted_vals = np.sort(np.abs(vals))[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sorted_vals, lw=1.5)
    ax.set_xlabel("Subbatch rank (sorted by |Aᵀb|)")
    ax.set_ylabel("|Aᵀ b| magnitude")
    ax.set_title(f"{title_prefix} projection spectrum (epoch {epoch})")
    fig.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fp = save_dir / f"epoch{epoch}_projection_spectrum.png"
    fig.savefig(fp, dpi=400)
    plt.close(fig)

# for OMP: top vs bottom k singleton quality
def plot_singleton_quality(top_cos, top_rr, bot_cos, bot_rr, save_dir, epoch=None, title_prefix="OMP", figsize=(12,8), plot_style='notebook'):
    """
    Scatter plot comparing singleton cosine vs residual ratio
    for top-k (selected) and bottom-k (least aligned) subbatches.
    """
    sns.set_context(plot_style)
    top_cos = np.asarray(top_cos, dtype=float)
    top_rr = np.asarray(top_rr, dtype=float)
    bot_cos = np.asarray(bot_cos, dtype=float)
    bot_rr = np.asarray(bot_rr, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(top_cos, top_rr, s=18, alpha=0.8, label="Top-k", marker="o")
    ax.scatter(bot_cos, bot_rr, s=18, alpha=0.8, label="Bottom-k", marker="x")
    ax.set_xlabel("Cosine(a_j, b)")
    ax.set_ylabel("Residual ratio ||b - α a_j|| / ||b||")
    ax.set_title(f"{title_prefix} singleton quality (epoch {epoch})")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fp = save_dir / f"epoch{epoch}_singleton_quality.png"
    fig.savefig(fp, dpi=400)
    plt.close(fig)