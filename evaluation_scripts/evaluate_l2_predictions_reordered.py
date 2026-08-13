import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

pickle_file = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/full_model/new_L2_focal_loss/test/results.pkl"

# Original class order — must match the integer indices stored in the pickle
ORIGINAL_CLASS_NAMES = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']

# Desired display order for the confusion matrix
DISPLAY_CLASS_NAMES = ['CADE27', 'ABCO', 'ABMA', 'PSME', 'NODE3', 'PICO', 'PILA', 'PIPJ']

# Indices into ORIGINAL_CLASS_NAMES for each class in DISPLAY_CLASS_NAMES
REORDER_INDICES = [ORIGINAL_CLASS_NAMES.index(cls) for cls in DISPLAY_CLASS_NAMES]


def load_predictions(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_confusion_matrix(predictions, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred in predictions:
        gt_idx = pred['gt_label'].item()
        pred_idx = pred['pred_label'].item()
        confusion[gt_idx, pred_idx] += 1
    return confusion


def plot_confusion_matrix(confusion_orig, output_path="l2_image_level_confusion_matrix_reordered.png"):
    # Reorder both rows (true) and columns (predicted) to the display order
    confusion = confusion_orig[np.ix_(REORDER_INDICES, REORDER_INDICES)]
    class_names = DISPLAY_CLASS_NAMES

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    # ax.set_title('Confusion Matrix - Image Counts', fontsize=14, fontweight='bold')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(confusion[i, j]),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                    fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 80)
    confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    print(confusion_df)
    print("=" * 80)


def main():
    print("Loading L2 predictions...")
    predictions = load_predictions(pickle_file)
    print(f"Loaded {len(predictions)} predictions")

    num_classes = len(ORIGINAL_CLASS_NAMES)
    num_classes_pkl = predictions[0]['num_classes']
    if num_classes_pkl != num_classes:
        raise ValueError(f"Pickle has {num_classes_pkl} classes but expected {num_classes}")

    confusion = create_confusion_matrix(predictions, num_classes)
    plot_confusion_matrix(confusion)


if __name__ == "__main__":
    main()
