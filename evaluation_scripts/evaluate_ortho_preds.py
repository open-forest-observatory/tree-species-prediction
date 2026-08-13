import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import re
import os

# Load your pickle file
pickle_file = '/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/full_model/new_L2_ortho_filtered/test/results.pkl'
# metadata_csv = '/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/test_metadata.csv'
metadata_csv = '/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_nadir_2/test_metadata.csv'

def load_predictions(pickle_file):
    """Load predictions from pickle file"""
    with open(pickle_file, 'rb') as f:
        predictions = pickle.load(f)
    return predictions

def load_valid_tree_set(metadata_csv):
    """
    Load the set of valid (dataset_name, tree_unique_id) combinations from the CSV.
    Returns a set of tuples: {(dataset_name, str(tree_unique_id)), ...}
    """
    df = pd.read_csv(metadata_csv)
    valid_set = set(
        zip(df['dataset_name'].astype(str), df['tree_unique_id'].astype(str))
    )
    print(f"Loaded {len(df)} rows from metadata CSV → {len(valid_set)} unique (dataset, tree_id) pairs")
    return valid_set

def parse_img_path(img_path):
    """
    Extract dataset_name and tree_unique_id from an img_path.

    Two known path formats:
      1. Ortho trees:
         .../cropped_ortho_trees/test/ABCO/0006_000142_000141_tree00133.png
         filename = "0006_000142_000141_tree00133.png"
         dataset_name  = "0006_000142_000141"   (everything before _tree)
         tree_unique_id = "133"                 (digits after 'tree', leading zeros stripped)

      2. Live trees:
         .../cropped_trees_live/0115_000676_000743/treeID00599/2124221919.png
         dataset_name  = "0115_000676_000743"   (parent folder of treeID folder)
         tree_unique_id = "599"                 (digits in treeID folder, leading zeros stripped)
    """
    filename = os.path.basename(img_path)
    parts = img_path.split('/')

    # Format 1: filename contains "_tree\d+"
    m = re.search(r'_tree(\d+)\.png$', filename, re.IGNORECASE)
    if m:
        tree_id = str(int(m.group(1)))  # strip leading zeros
        # dataset_name is the filename stem before "_tree\d+"
        dataset_name = re.sub(r'_tree\d+$', '', filename[:-4])  # remove .png then _treeXXX
        return dataset_name, tree_id

    # Format 2: path contains "treeID\d+" as a directory component
    for i, part in enumerate(parts):
        m = re.match(r'treeID(\d+)$', part, re.IGNORECASE)
        if m:
            tree_id = str(int(m.group(1)))  # strip leading zeros
            # dataset_name is the directory immediately before the treeID folder
            dataset_name = parts[i - 1] if i > 0 else ''
            return dataset_name, tree_id

    # Fallback: could not parse
    return None, None

def filter_predictions(predictions, valid_set):
    """Keep only predictions whose (dataset_name, tree_unique_id) is in valid_set."""
    filtered = []
    skipped = 0
    unparseable = 0

    for pred in predictions:
        img_path = pred.get('img_path', '')
        dataset_name, tree_id = parse_img_path(img_path)

        if dataset_name is None:
            unparseable += 1
            skipped += 1
            continue

        if (dataset_name, tree_id) in valid_set:
            filtered.append(pred)
        else:
            skipped += 1

    print(f"\nFiltering results:")
    print(f"  Total predictions:   {len(predictions)}")
    print(f"  Unparseable paths:   {unparseable}")
    print(f"  Kept (in CSV):       {len(filtered)}")
    print(f"  Skipped (not in CSV):{skipped - unparseable}")
    return filtered

def get_class_names(predictions):
    """Extract class names from predictions"""
    class_names = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']
    num_classes = predictions[0]['num_classes']

    if num_classes != len(class_names):
        print(f"Warning: Expected {len(class_names)} classes but found {num_classes}")
        class_names = [f"Class_{i}" for i in range(num_classes)]

    return class_names, num_classes

def calculate_accuracy_metrics(predictions, class_names):
    """Calculate overall and per-class accuracies (recall)"""
    total = len(predictions)
    correct = sum(1 for p in predictions if p['gt_label'].item() == p['pred_label'].item())
    overall_accuracy = correct / total if total > 0 else 0

    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pred in predictions:
        gt_idx = pred['gt_label'].item()
        pred_idx = pred['pred_label'].item()

        gt_class = class_names[gt_idx]
        class_stats[gt_class]['total'] += 1

        if gt_idx == pred_idx:
            class_stats[gt_class]['correct'] += 1

    class_accuracies = {}
    for cls, stats in class_stats.items():
        class_accuracies[cls] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    return overall_accuracy, class_accuracies, class_stats

def calculate_precision_metrics(predictions, class_names):
    """Calculate overall and per-class precision"""
    class_precision_stats = defaultdict(lambda: {'true_positives': 0, 'predicted_total': 0})

    for pred in predictions:
        gt_idx = pred['gt_label'].item()
        pred_idx = pred['pred_label'].item()

        pred_class = class_names[pred_idx]
        class_precision_stats[pred_class]['predicted_total'] += 1

        if gt_idx == pred_idx:
            class_precision_stats[pred_class]['true_positives'] += 1

    class_precisions = {}
    for cls in class_names:
        stats = class_precision_stats[cls]
        class_precisions[cls] = (
            stats['true_positives'] / stats['predicted_total']
            if stats['predicted_total'] > 0 else 0
        )

    overall_precision = np.mean([class_precisions[cls] for cls in class_names])

    return overall_precision, class_precisions, class_precision_stats

def create_confusion_matrix(predictions, num_classes):
    """Create confusion matrix"""
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for pred in predictions:
        gt_idx = pred['gt_label'].item()
        pred_idx = pred['pred_label'].item()
        confusion[gt_idx, pred_idx] += 1

    return confusion

def print_results(overall_accuracy, class_accuracies, class_stats, class_names,
                  overall_precision, class_precisions, class_precision_stats):
    """Print formatted results including recall and precision"""
    print("=" * 100)
    print("L2 MODEL EVALUATION RESULTS  (filtered to test_metadata.csv trees)")
    print("=" * 100)

    print(f"\nOverall Recall:    {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Overall Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)  [macro average]")
    print(f"Total samples: {sum(stats['total'] for stats in class_stats.values())}")

    print("\nPer-Class Results:")
    print("-" * 100)
    print(f"{'Class':<15} {'Recall':<12} {'Correct/Total':<20} {'Precision':<12} {'TP/Predicted':<20}")
    print("-" * 100)

    for cls in class_names:
        if cls in class_accuracies:
            rec = class_accuracies[cls]
            rec_stats = class_stats[cls]
            prec = class_precisions[cls]
            prec_stats = class_precision_stats[cls]
            print(
                f"{cls:<15} {rec:>6.2%}      "
                f"{rec_stats['correct']:>5} / {rec_stats['total']:<10}   "
                f"{prec:>6.2%}      "
                f"{prec_stats['true_positives']:>5} / {prec_stats['predicted_total']:<10}"
            )

    print("=" * 100)

def plot_confusion_matrix(confusion, class_names):
    """Plot confusion matrix with counts"""
    fig, ax = plt.subplots(figsize=(9, 9))

    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Image Counts\n(filtered to test_metadata.csv trees)',
                 fontsize=14, fontweight='bold')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(confusion[i, j]),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                    fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 80)
    confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    print(confusion_df)
    print("=" * 80)

def main():
    # Load metadata CSV and build valid set
    print("Loading metadata CSV...")
    valid_set = load_valid_tree_set(metadata_csv)

    # Load all predictions
    print("\nLoading predictions from pickle...")
    predictions = load_predictions(pickle_file)
    print(f"Loaded {len(predictions)} predictions")

    # Filter to only trees present in the CSV
    predictions = filter_predictions(predictions, valid_set)

    if len(predictions) == 0:
        print("ERROR: No predictions remain after filtering. Check path parsing logic.")
        return

    # Get class information
    class_names, num_classes = get_class_names(predictions)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Calculate metrics
    print("\nCalculating metrics...")
    overall_accuracy, class_accuracies, class_stats = calculate_accuracy_metrics(predictions, class_names)
    overall_precision, class_precisions, class_precision_stats = calculate_precision_metrics(predictions, class_names)

    # Create confusion matrix
    confusion = create_confusion_matrix(predictions, num_classes)

    # Print results
    print_results(overall_accuracy, class_accuracies, class_stats, class_names,
                  overall_precision, class_precisions, class_precision_stats)

    # Plot confusion matrix
    plot_confusion_matrix(confusion, class_names)

if __name__ == "__main__":
    main()