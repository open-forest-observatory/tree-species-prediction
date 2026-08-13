import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# This script generates tree-level confusion matrices for L2 and L4 models, with L2 classes grouped by their corresponding L4 classes.


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
L2_PKL_PATH      = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/full_model/new_L2_focal_loss/test/results.pkl"
L2_METADATA_PATH = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/test_metadata.csv"
CROSSWALK_PATH   = '/ofo-share/project-data/species-prediction-project/intermediate/species_class_crosswalk.csv'

L4_PKL_PATH      = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/full_model/new_l4_focal_loss/test/results.pkl"
L4_METADATA_PATH = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l4_new/test_metadata.csv"

OUTPUT_L2 = "/ofo-share/repos/amritha/tree-species-prediction/figures/cm_l2.png"
OUTPUT_L4 = "/ofo-share/repos/amritha/tree-species-prediction/figures/cm_l4.png"

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
L2_CLASSES = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']
L4_CLASSES = ['CUPR', 'FIR', 'HW', 'PI']

GROUP_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# -----------------------------------------------------------------------------
# Mapping helpers
# -----------------------------------------------------------------------------
def create_l2_to_l4_mapping(crosswalk, l2_classes):
    l2_to_l4 = {}
    for l2 in l2_classes:
        rows = crosswalk[crosswalk['species_code_l2'] == l2]
        l2_to_l4[l2] = rows.iloc[0]['species_code_l4']
    return l2_to_l4

def get_sorted_l2_classes(l2_classes, l2_to_l4, l4_classes):
    sorted_l2 = []
    for l4 in l4_classes:
        members = [l2 for l2 in l2_classes if l2_to_l4[l2] == l4]
        sorted_l2.extend(members)
    return sorted_l2

def get_l4_group_spans(sorted_l2, l2_to_l4, l4_classes):
    spans = {}
    for l4 in l4_classes:
        indices = [i for i, l2 in enumerate(sorted_l2) if l2_to_l4[l2] == l4]
        spans[l4] = (min(indices), max(indices))
    return spans

# -----------------------------------------------------------------------------
# Image-level predictions
# -----------------------------------------------------------------------------
def get_l2_image_preds(predictions):
    image_preds = {}
    for pred in predictions:
        filename = pred['img_path'].split('/')[-1]
        image_preds[filename] = {
            'gt_l2_idx':   pred['gt_label'].item(),
            'pred_l2_idx': pred['pred_label'].item(),
        }
    return image_preds

def get_l4_image_preds(predictions):
    image_preds = {}
    for pred in predictions:
        filename = pred['img_path'].split('/')[-1]
        image_preds[filename] = {
            'gt_l4_idx':   pred['gt_label'].item(),
            'pred_l4_idx': pred['pred_label'].item(),
        }
    return image_preds

# -----------------------------------------------------------------------------
# Tree-level aggregation via majority voting
# -----------------------------------------------------------------------------
def aggregate_to_tree_level(image_preds, metadata_df, key_gt, key_pred):
    df = metadata_df.copy()
    df['filename'] = df['image_path'].apply(lambda x: x.split('/')[-1])
    df['pred_idx'] = df['filename'].map(lambda x: image_preds.get(x, {}).get(key_pred, None))
    df['gt_idx']   = df['filename'].map(lambda x: image_preds.get(x, {}).get(key_gt,   None))
    df = df.dropna(subset=['pred_idx', 'gt_idx'])

    df['unique_tree_id'] = (
        df['dataset_name'].astype(str) + '_' + df['tree_unique_id'].astype(str)
    )

    tree_results = []
    for tree_id, group in df.groupby('unique_tree_id'):
        gt_idx = int(group['gt_idx'].iloc[0])
        pred_indices = group['pred_idx'].astype(int).tolist()
        pred_idx, _ = Counter(pred_indices).most_common(1)[0]
        tree_results.append({'gt_idx': gt_idx, 'pred_idx': pred_idx})

    return pd.DataFrame(tree_results)

# -----------------------------------------------------------------------------
# Build confusion matrix
# -----------------------------------------------------------------------------
def build_cm(tree_results, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for _, row in tree_results.iterrows():
        cm[row['gt_idx'], row['pred_idx']] += 1
    return cm

# -----------------------------------------------------------------------------
# Draw and save a single confusion matrix
# -----------------------------------------------------------------------------
def save_cm(cm, class_labels, output_path, group_spans=None, group_colors=None, l4_classes=None):
    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = cm.max() if cm.max() > 0 else 1
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text_color = 'white' if val > vmax * 0.55 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    color=text_color, fontsize=9,
                    fontweight='bold' if i == j else 'normal')

    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', size=0)

    if group_spans and group_colors and l4_classes:
        for l4, color in zip(l4_classes, group_colors):
            start, end = group_spans[l4]
            size = end - start + 1
            rect = plt.Rectangle(
                (start - 0.5, start - 0.5), size, size,
                linewidth=3, edgecolor=color, facecolor='none',
                zorder=5, linestyle='-'
            )
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Loading L2 model predictions...")
    l2_predictions = load_pkl(L2_PKL_PATH)
    l2_metadata    = pd.read_csv(L2_METADATA_PATH)
    crosswalk      = pd.read_csv(CROSSWALK_PATH)

    print("Loading L4 model predictions...")
    l4_predictions = load_pkl(L4_PKL_PATH)
    l4_metadata    = pd.read_csv(L4_METADATA_PATH)

    l2_to_l4    = create_l2_to_l4_mapping(crosswalk, L2_CLASSES)
    sorted_l2   = get_sorted_l2_classes(L2_CLASSES, l2_to_l4, L4_CLASSES)
    sorted_l2_to_idx = {l2: i for i, l2 in enumerate(sorted_l2)}
    group_spans = get_l4_group_spans(sorted_l2, l2_to_l4, L4_CLASSES)

    raw_l2_preds = get_l2_image_preds(l2_predictions)
    raw_l4_preds = get_l4_image_preds(l4_predictions)

    l2_preds_sorted = {
        fname: {
            'gt_l2_idx':   sorted_l2_to_idx[L2_CLASSES[v['gt_l2_idx']]],
            'pred_l2_idx': sorted_l2_to_idx[L2_CLASSES[v['pred_l2_idx']]],
        }
        for fname, v in raw_l2_preds.items()
    }

    print("Aggregating to tree level...")
    tree_l2 = aggregate_to_tree_level(l2_preds_sorted, l2_metadata, 'gt_l2_idx', 'pred_l2_idx')
    tree_l4 = aggregate_to_tree_level(raw_l4_preds,    l4_metadata, 'gt_l4_idx', 'pred_l4_idx')

    cm_l2 = build_cm(tree_l2, len(sorted_l2))
    cm_l4 = build_cm(tree_l4, len(L4_CLASSES))

    colors = GROUP_COLORS[:len(L4_CLASSES)]

    save_cm(
        cm_l2, sorted_l2, OUTPUT_L2,
        group_spans=group_spans,
        group_colors=colors,
        l4_classes=L4_CLASSES
    )

    save_cm(cm_l4, L4_CLASSES, OUTPUT_L4)

if __name__ == "__main__":
    main()
