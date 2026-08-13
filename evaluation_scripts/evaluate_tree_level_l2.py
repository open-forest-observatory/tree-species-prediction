"""
Generate tree-level confusion matrix from image-level predictions.
Aggregates multiple image predictions per tree using majority voting.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# File paths
PKL_PATH = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/full_model/new_L2_focal_loss/test/results.pkl"
METADATA_PATH = "/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/test_metadata.csv"

# Class labels (in index order)
CLASSES = [
    'ABCO',
    'ABMA',
    'CADE27',
    'NODE3',
    'PICO',
    'PILA',
    'PIPJ',
    'PSME',
]

def load_predictions(pkl_path):
    """Load predictions from pickle file."""
    with open(pkl_path, 'rb') as f:
        predictions = pickle.load(f)
    return predictions

def aggregate_tree_predictions(predictions, metadata_df):
    """
    Aggregate image-level predictions to tree-level using majority voting.
    
    Args:
        predictions: List of prediction dictionaries from pkl file
        metadata_df: DataFrame with tree metadata
        
    Returns:
        DataFrame with tree-level predictions and ground truth
    """
    # Create a mapping from filename to prediction (since paths differ)
    pred_map = {}
    for pred in predictions:
        img_path = pred['img_path']
        filename = img_path.split('/')[-1]  # Extract just the filename
        pred_label_idx = pred['pred_label'].item()
        gt_label_idx = pred['gt_label'].item()
        pred_map[filename] = {
            'pred_idx': pred_label_idx,
            'gt_idx': gt_label_idx
        }
    
    # Extract filename from metadata paths and match
    metadata_df['filename'] = metadata_df['image_path'].apply(lambda x: x.split('/')[-1])
    metadata_df['pred_idx'] = metadata_df['filename'].map(
        lambda x: pred_map.get(x, {}).get('pred_idx', None)
    )
    metadata_df['gt_idx'] = metadata_df['filename'].map(
        lambda x: pred_map.get(x, {}).get('gt_idx', None)
    )
    
    # Remove rows without predictions
    metadata_df = metadata_df.dropna(subset=['pred_idx', 'gt_idx'])
    
    # Create unique tree ID
    metadata_df['unique_tree_id'] = (
        metadata_df['dataset_name'].astype(str) + '_' + 
        metadata_df['tree_unique_id'].astype(str)
    )
    
    # Group by tree and aggregate predictions
    tree_results = []
    
    for tree_id, group in metadata_df.groupby('unique_tree_id'):
        # Get ground truth (should be same for all images of a tree)
        gt_species = group['species_l2'].iloc[0]
        gt_idx = group['gt_idx'].iloc[0]
        
        # Majority vote for predictions
        pred_indices = group['pred_idx'].astype(int).tolist()
        pred_idx, _ = Counter(pred_indices).most_common(1)[0]
        
        tree_results.append({
            'tree_id': tree_id,
            'gt_species': gt_species,
            'gt_idx': int(gt_idx),
            'pred_idx': pred_idx,
            'pred_species': CLASSES[pred_idx],
            'num_images': len(group)
        })
    
    return pd.DataFrame(tree_results)

def create_confusion_matrix(tree_results):
    """
    Create confusion matrix from tree-level results.
    
    Args:
        tree_results: DataFrame with tree-level predictions
        
    Returns:
        Confusion matrix as numpy array
    """
    n_classes = len(CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for _, row in tree_results.iterrows():
        gt_idx = row['gt_idx']
        pred_idx = row['pred_idx']
        cm[gt_idx, pred_idx] += 1
    
    return cm

def plot_confusion_matrix(cm, classes):
    """
    Plot confusion matrix with counts using matplotlib only.
    
    Args:
        cm: Confusion matrix (numpy array)
        classes: List of class labels
    """
    # Calculate recall per class
    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)
    class_recalls = class_correct / class_totals

    # Calculate precision per class
    pred_totals = cm.sum(axis=0)
    class_precisions = np.where(pred_totals > 0, class_correct / pred_totals, 0)

    # Overall recall and precision (macro averages)
    overall_recall = class_correct.sum() / class_totals.sum()
    overall_precision = class_precisions.mean()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot heatmap using imshow
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add counts as text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm[i, j]
            
            # Choose text color based on background
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            
            # Add text
            ax.text(j, i, str(count),
                   ha='center', va='center',
                   color=text_color,
                   fontsize=11,
                   weight='bold' if i == j else 'normal')
    
    # Labels and title
    ax.set_xlabel('Predicted Class', fontsize=12, weight='bold')
    ax.set_ylabel('True Class', fontsize=12, weight='bold')
    ax.set_title(
        f'Confusion Matrix - Tree Counts',
        fontsize=14,
        weight='bold',
        pad=20
    )
    
    # Add grid
    ax.set_xticks(np.arange(len(classes)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(classes)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    # Print per-class statistics
    print("\n" + "="*70)
    print("TREE-LEVEL CLASSIFICATION RESULTS")
    print("="*70)
    print(f"\nOverall Recall:    {overall_recall:.2%}")
    print(f"Overall Precision: {overall_precision:.2%}  [macro average]")
    print(f"Total Trees: {class_totals.sum()}")
    print("\nPer-Class Performance:")
    print("-"*70)
    print(f"{'Class':<10} {'Recall':<10} {'Correct/Total':<20} {'Precision':<10} {'TP/Predicted':<20}")
    print("-"*70)
    
    for i, cls in enumerate(classes):
        print(
            f"{cls:<10} {class_recalls[i]:>8.2%}   "
            f"{class_correct[i]:<4} / {class_totals[i]:<12}   "
            f"{class_precisions[i]:>8.2%}   "
            f"{class_correct[i]:<4} / {pred_totals[i]:<12}"
        )
    
    print("="*70 + "\n")
    
    return fig

def main():
    """Main execution function."""
    print("Loading predictions...")
    predictions = load_predictions(PKL_PATH)
    print(f"Loaded {len(predictions)} image predictions")
    
    print("\nLoading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata for {len(metadata)} images")
    
    print("\nAggregating to tree level...")
    tree_results = aggregate_tree_predictions(predictions, metadata)

    # keep rows whose unique_tree_id is NOT in exclusion_list
    exclusion_list = [
        "0048_000173_000171_356",
        "0048_000174_000171_521",
        "0049_000136_000133_411",
        "0068_000434_000440_1204",
        "0068_000434_000440_1461",
        "0068_000434_000440_952",
        "0110_000136_000133_615",
        "0115_000645_000743_1629",
        "0115_000645_000743_1880",
        "0115_000645_000743_882",
        "0115_000676_000743_1037"
    ]
    excluded = tree_results['tree_id'].isin(exclusion_list).sum()
    tree_results = tree_results[~tree_results['tree_id'].isin(exclusion_list)]
    print(f"Excluded {excluded} trees missing nadir views. Remaining: {len(tree_results)}")

    print(f"Aggregated to {len(tree_results)} unique trees")
    
    print("\nCreating confusion matrix...")
    cm = create_confusion_matrix(tree_results)
    
    print("\nPlotting confusion matrix...")
    fig = plot_confusion_matrix(cm, CLASSES)
    
    plt.show()
    
    # Return results for further analysis if needed
    return tree_results, cm

if __name__ == "__main__":
    tree_results, cm = main()

