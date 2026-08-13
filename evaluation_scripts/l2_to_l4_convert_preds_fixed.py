# Tree-level evaluation

import pickle
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
pickle_file = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/full_model/new_L2_focal_loss/test/results.pkl"
crosswalk_file = '/ofo-share/project-data/species-prediction-project/intermediate/species_class_crosswalk.csv'
metadata_file = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/test_metadata.csv"

# -----------------------------------------------------------------------------
# L2 classes (model order)
# -----------------------------------------------------------------------------
l2_classes = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_data(pickle_file, crosswalk_file, metadata_file):
    with open(pickle_file, 'rb') as f:
        predictions = pickle.load(f)
    crosswalk = pd.read_csv(crosswalk_file)
    metadata = pd.read_csv(metadata_file)
    return predictions, crosswalk, metadata

# -----------------------------------------------------------------------------
# Mapping helpers
# -----------------------------------------------------------------------------
def create_l2_to_l4_mapping(crosswalk, l2_classes):
    l2_to_l4 = {}
    for l2 in l2_classes:
        rows = crosswalk[crosswalk['species_code_l2'] == l2]
        l2_to_l4[l2] = rows.iloc[0]['species_code_l4']
    return l2_to_l4

def get_l4_to_l2_mapping(l2_to_l4):
    l4_to_l2 = defaultdict(list)
    for l2, l4 in l2_to_l4.items():
        l4_to_l2[l4].append(l2)
    return dict(l4_to_l2)

def get_l4_classes(l2_to_l4):
    l4_classes = sorted(set(l2_to_l4.values()))
    l4_to_idx = {c: i for i, c in enumerate(l4_classes)}
    return l4_classes, l4_to_idx

# -----------------------------------------------------------------------------
# Image-level L4 prediction (KEY LOGIC)
# -----------------------------------------------------------------------------
def get_image_l4_predictions(
    predictions, l2_classes, l2_to_l4, l4_to_idx, l4_to_l2
):
    """Map each image's L2 prediction to L4, returning a dict keyed by filename."""
    image_l4_preds = {}

    for pred in predictions:
        filename = pred['img_path'].split('/')[-1]
        l2_scores = pred['pred_score'].numpy()

        # ---- Ground truth ----
        gt_l2_idx = pred['gt_label'].item()
        gt_l2_code = l2_classes[gt_l2_idx]
        gt_l4_code = l2_to_l4[gt_l2_code]
        gt_l4_idx = l4_to_idx[gt_l4_code]

        # ---- Prediction ----
        pred_l2_idx = pred['pred_label'].item()
        pred_l2_code = l2_classes[pred_l2_idx]
        pred_l4_code = l2_to_l4[pred_l2_code]

        # Single-mapped L4 → trust L2 argmax
        if len(l4_to_l2[pred_l4_code]) == 1:
            pred_l4_idx = l4_to_idx[pred_l4_code]
        # Multi-mapped L4 → sum probabilities
        else:
            l4_scores = np.zeros(len(l4_to_idx))
            for i, l2 in enumerate(l2_classes):
                l4 = l2_to_l4[l2]
                l4_scores[l4_to_idx[l4]] += l2_scores[i]
            pred_l4_idx = int(np.argmax(l4_scores))

        image_l4_preds[filename] = {
            'gt_l4_idx': gt_l4_idx,
            'pred_l4_idx': pred_l4_idx,
        }

    return image_l4_preds

# -----------------------------------------------------------------------------
# Tree-level aggregation via majority voting
# -----------------------------------------------------------------------------
def aggregate_to_tree_level(image_l4_preds, metadata_df, l4_classes, l4_to_idx):
    """Aggregate image-level L4 predictions to tree level using majority voting."""
    metadata_df = metadata_df.copy()
    metadata_df['filename'] = metadata_df['image_path'].apply(lambda x: x.split('/')[-1])
    metadata_df['pred_l4_idx'] = metadata_df['filename'].map(
        lambda x: image_l4_preds.get(x, {}).get('pred_l4_idx', None)
    )
    metadata_df['gt_l4_idx'] = metadata_df['filename'].map(
        lambda x: image_l4_preds.get(x, {}).get('gt_l4_idx', None)
    )

    metadata_df = metadata_df.dropna(subset=['pred_l4_idx', 'gt_l4_idx'])

    metadata_df['unique_tree_id'] = (
        metadata_df['dataset_name'].astype(str) + '_' +
        metadata_df['tree_unique_id'].astype(str)
    )

    tree_results = []
    for tree_id, group in metadata_df.groupby('unique_tree_id'):
        gt_l4_idx = int(group['gt_l4_idx'].iloc[0])

        # Majority vote over L4 predictions
        pred_indices = group['pred_l4_idx'].astype(int).tolist()
        pred_l4_idx, _ = Counter(pred_indices).most_common(1)[0]

        tree_results.append({
            'tree_id': tree_id,
            'gt_l4_idx': gt_l4_idx,
            'gt_l4_code': l4_classes[gt_l4_idx],
            'pred_l4_idx': pred_l4_idx,
            'pred_l4_code': l4_classes[pred_l4_idx],
            'num_images': len(group),
        })

    return pd.DataFrame(tree_results)

# -----------------------------------------------------------------------------
# Metrics (recall + precision)
# -----------------------------------------------------------------------------
def calculate_metrics(tree_results, l4_classes):
    """Calculate per-class and overall recall and precision."""
    recall_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    precision_stats = defaultdict(lambda: {'true_positives': 0, 'predicted_total': 0})

    for _, row in tree_results.iterrows():
        gt = l4_classes[row['gt_l4_idx']]
        pred = l4_classes[row['pred_l4_idx']]

        recall_stats[gt]['total'] += 1
        precision_stats[pred]['predicted_total'] += 1

        if row['gt_l4_idx'] == row['pred_l4_idx']:
            recall_stats[gt]['correct'] += 1
            precision_stats[pred]['true_positives'] += 1

    class_recall = {
        c: recall_stats[c]['correct'] / recall_stats[c]['total']
        if recall_stats[c]['total'] > 0 else 0
        for c in l4_classes
    }
    class_precision = {
        c: precision_stats[c]['true_positives'] / precision_stats[c]['predicted_total']
        if precision_stats[c]['predicted_total'] > 0 else 0
        for c in l4_classes
    }

    overall_recall = (
        sum(recall_stats[c]['correct'] for c in l4_classes) /
        sum(recall_stats[c]['total'] for c in l4_classes)
    )
    overall_precision = np.mean([class_precision[c] for c in l4_classes])

    return overall_recall, class_recall, recall_stats, overall_precision, class_precision, precision_stats

# -----------------------------------------------------------------------------
# Print results
# -----------------------------------------------------------------------------
def print_results(overall_recall, class_recall, recall_stats,
                  overall_precision, class_precision, precision_stats, l4_classes):
    total_trees = sum(recall_stats[c]['total'] for c in l4_classes)

    print("\n" + "=" * 75)
    print("TREE-LEVEL L4 CLASSIFICATION RESULTS (L2 → L4 mapped)")
    print("=" * 75)
    print(f"\nOverall Recall:    {overall_recall:.2%}")
    print(f"Overall Precision: {overall_precision:.2%}  [macro average]")
    print(f"Total Trees: {total_trees}")
    print("\nPer-Class Performance:")
    print("-" * 75)
    print(f"{'Class':<10} {'Recall':<10} {'Correct/Total':<20} {'Precision':<10} {'TP/Predicted':<20}")
    print("-" * 75)

    for c in l4_classes:
        rs = recall_stats[c]
        ps = precision_stats[c]
        print(
            f"{c:<10} {class_recall[c]:>8.2%}   "
            f"{rs['correct']:<4} / {rs['total']:<12}   "
            f"{class_precision[c]:>8.2%}   "
            f"{ps['true_positives']:<4} / {ps['predicted_total']:<12}"
        )

    print("=" * 75 + "\n")

# -----------------------------------------------------------------------------
# Confusion matrix
# -----------------------------------------------------------------------------
def create_confusion_matrix(tree_results, l4_classes):
    n = len(l4_classes)
    confusion = np.zeros((n, n), dtype=int)
    for _, row in tree_results.iterrows():
        confusion[row['gt_l4_idx'], row['pred_l4_idx']] += 1
    return confusion

def plot_confusion_matrix(confusion, l4_classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Raw counts ----
    im1 = ax1.imshow(confusion, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(len(l4_classes)))
    ax1.set_yticks(range(len(l4_classes)))
    ax1.set_xticklabels(l4_classes)
    ax1.set_yticklabels(l4_classes)
    ax1.set_xlabel('Predicted Class', fontweight='bold')
    ax1.set_ylabel('True Class', fontweight='bold')
    ax1.set_title('Confusion Matrix - Tree Counts', fontweight='bold')

    for i in range(len(l4_classes)):
        for j in range(len(l4_classes)):
            ax1.text(
                j, i, confusion[i, j],
                ha="center", va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black"
            )

    plt.colorbar(im1, ax=ax1, label='Count')

    # ---- Normalized by true class ----
    norm = confusion.astype(float) / confusion.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    im2 = ax2.imshow(norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(l4_classes)))
    ax2.set_yticks(range(len(l4_classes)))
    ax2.set_xticklabels(l4_classes)
    ax2.set_yticklabels(l4_classes)
    ax2.set_xlabel('Predicted Class', fontweight='bold')
    ax2.set_ylabel('True Class', fontweight='bold')
    ax2.set_title('Confusion Matrix (Row-normalized)', fontweight='bold')

    for i in range(len(l4_classes)):
        for j in range(len(l4_classes)):
            ax2.text(
                j, i, f"{norm[i, j]:.1%}",
                ha="center", va="center",
                color="white" if norm[i, j] > 0.5 else "black"
            )

    plt.colorbar(im2, ax=ax2, label='Fraction')
    plt.tight_layout()
    plt.show()

    print("\nCONFUSION MATRIX (rows=true, cols=predicted)")
    print(pd.DataFrame(confusion, index=l4_classes, columns=l4_classes))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    predictions, crosswalk, metadata = load_data(pickle_file, crosswalk_file, metadata_file)
    print(f"Loaded {len(predictions)} image predictions")
    print(f"Loaded metadata for {len(metadata)} images")

    l2_to_l4 = create_l2_to_l4_mapping(crosswalk, l2_classes)
    l4_to_l2 = get_l4_to_l2_mapping(l2_to_l4)
    l4_classes, l4_to_idx = get_l4_classes(l2_to_l4)

    print("\nL4 → L2 structure:")
    for k, v in l4_to_l2.items():
        print(f"  {k}: {v}")

    # Map image predictions to L4
    image_l4_preds = get_image_l4_predictions(
        predictions, l2_classes, l2_to_l4, l4_to_idx, l4_to_l2
    )

    # Aggregate to tree level
    print("\nAggregating to tree level...")
    tree_results = aggregate_to_tree_level(image_l4_preds, metadata, l4_classes, l4_to_idx)
    print(f"Aggregated to {len(tree_results)} unique trees")

    # Calculate metrics
    (overall_recall, class_recall, recall_stats,
     overall_precision, class_precision, precision_stats) = calculate_metrics(tree_results, l4_classes)

    # Print results
    print_results(overall_recall, class_recall, recall_stats,
                  overall_precision, class_precision, precision_stats, l4_classes)

    # Confusion matrix
    confusion = create_confusion_matrix(tree_results, l4_classes)
    plot_confusion_matrix(confusion, l4_classes)

if __name__ == "__main__":
    main()


# import pickle
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# import matplotlib.pyplot as plt

# # -----------------------------------------------------------------------------
# # Paths
# # -----------------------------------------------------------------------------
# pickle_file = '/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/yuba_model/fixed_dataset/new_L2_focal_loss/20260206_193532/test/results.pkl'
# crosswalk_file = '/ofo-share/project-data/species-prediction-project/intermediate/species_class_crosswalk.csv'

# # -----------------------------------------------------------------------------
# # L2 classes (model order)
# # -----------------------------------------------------------------------------
# l2_classes = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']

# # -----------------------------------------------------------------------------
# # Loading
# # -----------------------------------------------------------------------------
# def load_data(pickle_file, crosswalk_file):
#     with open(pickle_file, 'rb') as f:
#         predictions = pickle.load(f)
#     crosswalk = pd.read_csv(crosswalk_file)
#     return predictions, crosswalk

# # -----------------------------------------------------------------------------
# # Mapping helpers
# # -----------------------------------------------------------------------------
# def create_l2_to_l4_mapping(crosswalk, l2_classes):
#     l2_to_l4 = {}
#     for l2 in l2_classes:
#         rows = crosswalk[crosswalk['species_code_l2'] == l2]
#         l2_to_l4[l2] = rows.iloc[0]['species_code_l4']
#     return l2_to_l4

# def get_l4_to_l2_mapping(l2_to_l4):
#     l4_to_l2 = defaultdict(list)
#     for l2, l4 in l2_to_l4.items():
#         l4_to_l2[l4].append(l2)
#     return dict(l4_to_l2)

# def get_l4_classes(l2_to_l4):
#     l4_classes = sorted(set(l2_to_l4.values()))
#     l4_to_idx = {c: i for i, c in enumerate(l4_classes)}
#     return l4_classes, l4_to_idx

# # -----------------------------------------------------------------------------
# # Hybrid aggregation (KEY LOGIC)
# # -----------------------------------------------------------------------------
# def aggregate_predictions_to_l4(
#     predictions, l2_classes, l2_to_l4, l4_to_idx, l4_to_l2
# ):
#     l4_predictions = []

#     for pred in predictions:
#         l2_scores = pred['pred_score'].numpy()

#         # ---- Ground truth ----
#         gt_l2_idx = pred['gt_label'].item()
#         gt_l2_code = l2_classes[gt_l2_idx]
#         gt_l4_code = l2_to_l4[gt_l2_code]
#         gt_l4_idx = l4_to_idx[gt_l4_code]

#         # ---- Prediction ----
#         pred_l2_idx = pred['pred_label'].item()
#         pred_l2_code = l2_classes[pred_l2_idx]
#         pred_l4_code = l2_to_l4[pred_l2_code]

#         # Single-mapped L4 → trust L2 argmax
#         if len(l4_to_l2[pred_l4_code]) == 1:
#             pred_l4_idx = l4_to_idx[pred_l4_code]

#         # Multi-mapped L4 → sum probabilities
#         else:
#             l4_scores = np.zeros(len(l4_to_idx))
#             for i, l2 in enumerate(l2_classes):
#                 l4 = l2_to_l4[l2]
#                 l4_scores[l4_to_idx[l4]] += l2_scores[i]
#             pred_l4_idx = np.argmax(l4_scores)

#         l4_predictions.append({
#             'gt_l4_idx': gt_l4_idx,
#             'pred_l4_idx': pred_l4_idx,
#             'gt_l4_code': gt_l4_code,
#             'original_l2_pred': pred_l2_code,
#         })

#     return l4_predictions

# # -----------------------------------------------------------------------------
# # Metrics
# # -----------------------------------------------------------------------------
# def calculate_accuracy_metrics(l4_predictions, l4_classes):
#     class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

#     for p in l4_predictions:
#         gt = l4_classes[p['gt_l4_idx']]
#         class_stats[gt]['total'] += 1
#         if p['gt_l4_idx'] == p['pred_l4_idx']:
#             class_stats[gt]['correct'] += 1

#     class_acc = {
#         c: v['correct'] / v['total']
#         for c, v in class_stats.items()
#     }

#     overall = sum(v['correct'] for v in class_stats.values()) / \
#               sum(v['total'] for v in class_stats.values())

#     return overall, class_acc, class_stats

# # -----------------------------------------------------------------------------
# # Confusion matrix
# # -----------------------------------------------------------------------------
# def create_confusion_matrix(l4_predictions, l4_classes):
#     n = len(l4_classes)
#     confusion = np.zeros((n, n), dtype=int)

#     for p in l4_predictions:
#         confusion[p['gt_l4_idx'], p['pred_l4_idx']] += 1

#     return confusion

# def plot_confusion_matrix(confusion, l4_classes):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#     # ---- Raw counts ----
#     im1 = ax1.imshow(confusion, cmap='Blues', aspect='auto')
#     ax1.set_xticks(range(len(l4_classes)))
#     ax1.set_yticks(range(len(l4_classes)))
#     ax1.set_xticklabels(l4_classes)
#     ax1.set_yticklabels(l4_classes)
#     ax1.set_xlabel('Predicted Class', fontweight='bold')
#     ax1.set_ylabel('True Class', fontweight='bold')
#     ax1.set_title('Confusion Matrix - Image Counts', fontweight='bold')

#     for i in range(len(l4_classes)):
#         for j in range(len(l4_classes)):
#             ax1.text(
#                 j, i, confusion[i, j],
#                 ha="center", va="center",
#                 color="white" if confusion[i, j] > confusion.max() / 2 else "black"
#             )

#     plt.colorbar(im1, ax=ax1, label='Count')

#     # ---- Normalized by true class ----
#     norm = confusion.astype(float) / confusion.sum(axis=1, keepdims=True)
#     norm = np.nan_to_num(norm)

#     im2 = ax2.imshow(norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
#     ax2.set_xticks(range(len(l4_classes)))
#     ax2.set_yticks(range(len(l4_classes)))
#     ax2.set_xticklabels(l4_classes)
#     ax2.set_yticklabels(l4_classes)
#     ax2.set_xlabel('Predicted Class', fontweight='bold')
#     ax2.set_ylabel('True Class', fontweight='bold')
#     ax2.set_title('Confusion Matrix (Row-normalized)', fontweight='bold')

#     for i in range(len(l4_classes)):
#         for j in range(len(l4_classes)):
#             ax2.text(
#                 j, i, f"{norm[i, j]:.1%}",
#                 ha="center", va="center",
#                 color="white" if norm[i, j] > 0.5 else "black"
#             )

#     plt.colorbar(im2, ax=ax2, label='Fraction')
#     plt.tight_layout()
#     plt.show()

#     print("\nCONFUSION MATRIX (rows=true, cols=predicted)")
#     print(pd.DataFrame(confusion, index=l4_classes, columns=l4_classes))

# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------
# def main():
#     predictions, crosswalk = load_data(pickle_file, crosswalk_file)

#     l2_to_l4 = create_l2_to_l4_mapping(crosswalk, l2_classes)
#     l4_to_l2 = get_l4_to_l2_mapping(l2_to_l4)
#     l4_classes, l4_to_idx = get_l4_classes(l2_to_l4)

#     print("\nL4 → L2 structure:")
#     for k, v in l4_to_l2.items():
#         print(f"  {k}: {v}")

#     l4_predictions = aggregate_predictions_to_l4(
#         predictions,
#         l2_classes,
#         l2_to_l4,
#         l4_to_idx,
#         l4_to_l2
#     )

#     overall, class_acc, class_stats = calculate_accuracy_metrics(
#         l4_predictions, l4_classes
#     )

#     print("\nPer-class L4 accuracy:")
#     for c in l4_classes:
#         s = class_stats[c]
#         print(f"{c:5s}: {s['correct']} / {s['total']} ({class_acc[c]*100:.2f}%)")

#     print(f"\nOverall L4 accuracy: {overall*100:.2f}%")

#     confusion = create_confusion_matrix(l4_predictions, l4_classes)
#     plot_confusion_matrix(confusion, l4_classes)

# if __name__ == "__main__":
#     main()
