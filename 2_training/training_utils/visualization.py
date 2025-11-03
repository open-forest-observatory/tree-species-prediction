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