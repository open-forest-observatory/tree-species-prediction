from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def confusion_matrix(unique_class_labels, dataloader, model, device, exclude_empty=False):
    """Create a confusion matrix of ground truth labels vs predicted labels for all classes."""
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(unique_class_labels)).to(device)

    all_preds = []
    all_targets = []
    for imgs, labels, _ in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cm = confmat(all_preds, all_targets).cpu().numpy()

    # optionally remove rows/cols with no entries
    # typically if classes are excluded in training but loaded in the full dataset
    if exclude_empty: 
        mask = ~(cm.sum(axis=0) == 0) & ~(cm.sum(axis=1) == 0)
        cm = cm[np.ix_(mask, mask)]
        unique_class_labels = [lbl for lbl, keep in zip(unique_class_labels, mask) if keep]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_class_labels, yticklabels=unique_class_labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    return fig

def plot_sample_images(dset, title_prefix=''):
    shuffled_idxs = list(range(len(dset)))
    np.random.shuffle(shuffled_idxs)
    
    fig, axes = plt.subplots(4,4, figsize=(16,16))
    for i, ax in enumerate(axes.flatten()):
        sample = dset[shuffled_idxs[i]]
        img_reshape = sample[0].permute(1,2,0)
        label_str = dset.idx2label_map[sample[1]]

        ax.imshow(img_reshape)
        ax.set_title(f"{title_prefix} - {label_str}")
    
    fig.savefig(f"2_training/ckpts/{title_prefix}-sample_imgs.png")


# **********************************************************************
# 
# For Data Reduction / OMP Insights
# 
# **********************************************************************

# for OMP: sorted projection spectrum
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


# for OMP: Scatter vs subbatch index
def plot_projection_scatter(proj_vals, save_dir, epoch=None, title_prefix="OMP", figsize=(12,8), plot_style='notebook'):
    """
    Scatter plot of |Aᵀb| vs subbatch index.
    Reveals if certain dataset regions dominate the projection magnitude.
    """
    sns.set_context(plot_style)
    vals = np.asarray(proj_vals, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(np.arange(len(vals)), np.abs(vals), s=5, alpha=0.6)
    ax.set_xlabel("Subbatch index")
    ax.set_ylabel("|Aᵀ b|")
    ax.set_title(f"{title_prefix} projection scatter (epoch {epoch})")
    fig.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fp = save_dir / f"epoch{epoch}_projection_scatter.png"
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