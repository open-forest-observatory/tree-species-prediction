from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def confusion_matrix(unique_class_labels, dataloader, model, device):
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

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_class_labels, yticklabels=unique_class_labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Validation Confusion Matrix")
    return fig