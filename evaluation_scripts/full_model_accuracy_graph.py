#!/usr/bin/env python3
"""
Plot image-level vs tree-level accuracies for Full model only.
"""

import matplotlib.pyplot as plt
import numpy as np

# Class names
CLASSES = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']

# Full Model Accuracies
full_image_acc = [85.55, 66.62, 75.75, 34.15, 45.74, 84.51, 90.94, 91.30]
full_tree_acc = [89.20, 74.19, 81.17, 40.54, 71.43, 86.09, 94.10, 92.86]

# Create figure with one subplot
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

# Set up bar positions
x = np.arange(len(CLASSES))
width = 0.35

# Colors
image_color = '#3498db'  # Blue
tree_color = '#e74c3c'   # Red

# ========== FULL MODEL PLOT ==========
bars1_full = ax1.bar(x - width/2, full_image_acc, width, label='Image-level', 
                     color=image_color, alpha=0.8, edgecolor='black', linewidth=1)
bars2_full = ax1.bar(x + width/2, full_tree_acc, width, label='Tree-level', 
                     color=tree_color, alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on top of bars for Full Model
for bar in bars1_full:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2_full:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize plot
ax1.set_xlabel('Species', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Full Model performance on complete test set', 
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(CLASSES, fontsize=11)
ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 105)
ax1.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add overall title
fig.suptitle('Image-level vs Tree-level Classification Accuracy', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('/ofo-share/repos/amritha/tree-species-prediction/evaluation_scripts/full_model_accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("ACCURACY IMPROVEMENT SUMMARY (Tree - Image)")
print("="*70)
print(f"\n{'Class':<10} {'Full Model':<15}")
print("-"*70)
for i, cls in enumerate(CLASSES):
    full_improvement = full_tree_acc[i] - full_image_acc[i]
    print(f"{cls:<10} {full_improvement:>+6.2f}%")

print("\n" + "="*70)
avg_full_improvement = np.mean([full_tree_acc[i] - full_image_acc[i] for i in range(len(CLASSES))])
print(f"Average improvement:")
print(f"  Full Model: {avg_full_improvement:+.2f}%")
print("="*70 + "\n")