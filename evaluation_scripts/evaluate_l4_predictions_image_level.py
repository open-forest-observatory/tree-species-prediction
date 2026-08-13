import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import matplotlib.pyplot as plt

# Load your pickle file
pickle_file = '/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/yuba_model/fixed_dataset/new_L4_base/20260207_000501/test/results.pkl'

def load_predictions(pickle_file):
    """Load predictions from pickle file"""
    with open(pickle_file, 'rb') as f:
        predictions = pickle.load(f)
    return predictions

def get_class_names(predictions):
    """Extract class names from predictions"""
    # L4 class names
    class_names = ['CUPR', 'FIR', 'HW', 'PI']
    num_classes = predictions[0]['num_classes']
    
    # Verify we have the right number of classes
    if num_classes != len(class_names):
        print(f"Warning: Expected {len(class_names)} classes but found {num_classes}")
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    return class_names, num_classes

def calculate_accuracy_metrics(predictions, class_names):
    """Calculate overall and per-class accuracies"""
    total = len(predictions)
    correct = sum(1 for p in predictions if p['gt_label'].item() == p['pred_label'].item())
    overall_accuracy = correct / total if total > 0 else 0
    
    # Per-class metrics
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        gt_idx = pred['gt_label'].item()
        pred_idx = pred['pred_label'].item()
        
        gt_class = class_names[gt_idx]
        class_stats[gt_class]['total'] += 1
        
        if gt_idx == pred_idx:
            class_stats[gt_class]['correct'] += 1
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for cls, stats in class_stats.items():
        class_accuracies[cls] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    return overall_accuracy, class_accuracies, class_stats

def create_confusion_matrix(predictions, num_classes):
    """Create confusion matrix"""
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for pred in predictions:
        gt_idx = pred['gt_label'].item()
        pred_idx = pred['pred_label'].item()
        confusion[gt_idx, pred_idx] += 1
    
    return confusion

def print_results(overall_accuracy, class_accuracies, class_stats, class_names):
    """Print formatted results"""
    print("=" * 80)
    print("L4 MODEL EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Total samples: {sum(stats['total'] for stats in class_stats.values())}")
    
    print("\nPer-Class Results:")
    print("-" * 80)
    print(f"{'Class':<15} {'Accuracy':<12} {'Correct':<10} {'Total':<10} {'Error Rate':<12}")
    print("-" * 80)
    
    for cls in class_names:
        if cls in class_accuracies:
            acc = class_accuracies[cls]
            stats = class_stats[cls]
            error_rate = 1 - acc
            print(f"{cls:<15} {acc:>6.2%}      {stats['correct']:>5} / {stats['total']:<5}   {error_rate:>6.2%}")
    
    print("=" * 80)

def plot_confusion_matrix(confusion, class_names):
    """Plot confusion matrix with counts"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot raw counts
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Image Counts', fontsize=14, fontweight='bold')
    
    # Add text annotations for counts
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, int(confusion[i, j]),
                           ha="center", va="center", 
                           color="white" if confusion[i, j] > confusion.max()/2 else "black",
                           fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.show()
    
    # Print confusion matrix as table
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 80)
    confusion_df = pd.DataFrame(
        confusion, 
        index=class_names,
        columns=class_names
    )
    print(confusion_df)
    print("=" * 80)

def main():
    # Load predictions
    print("Loading L4 predictions...")
    predictions = load_predictions(pickle_file)
    print(f"Loaded {len(predictions)} predictions")
    
    # Get class information
    class_names, num_classes = get_class_names(predictions)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    overall_accuracy, class_accuracies, class_stats = calculate_accuracy_metrics(predictions, class_names)
    
    # Create confusion matrix
    confusion = create_confusion_matrix(predictions, num_classes)
    
    # Print results
    print_results(overall_accuracy, class_accuracies, class_stats, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion, class_names)

if __name__ == "__main__":
    main()