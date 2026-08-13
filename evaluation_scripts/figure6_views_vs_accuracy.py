import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

METADATA_PATH = '/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/datasets/full_mmpretrain_dataset_l2_new/test_metadata.csv'
RESULTS_PATH  = '/ofo-share/scratch/amritha/tree-species-scratch/january-run-2/checkpoints/full_model/new_L2_focal_loss/test/results.pkl'

CLASSES = ['ABCO', 'ABMA', 'CADE27', 'NODE3', 'PICO', 'PILA', 'PIPJ', 'PSME']
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))

STYLE = {
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
}
plt.rcParams.update(STYLE)

print("Loading metadata …")
meta = pd.read_csv(METADATA_PATH)
meta = meta[meta['species_l2'].isin(CLASSES)].reset_index(drop=True)

print("Loading results pickle …")
with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

print(f"  {len(results)} predictions loaded")

rows = []
for r in results:
    img_name   = os.path.basename(r['img_path'])
    gt_label   = int(r['gt_label'].item())
    pred_label = int(r['pred_label'].item())
    correct    = int(gt_label == pred_label)
    rows.append({'img_name': img_name, 'gt_label': gt_label,
                 'pred_label': pred_label, 'correct': correct})

pred_df = pd.DataFrame(rows)

meta['img_name'] = meta['image_path'].apply(os.path.basename)
df = pred_df.merge(
    meta[['img_name', 'tree_unique_id', 'dataset_name', 'species_l2']],
    on='img_name', how='left'
)
df['gt_class'] = df['species_l2']

TREE_KEY = ['dataset_name', 'tree_unique_id']

tree_df = (
    df.groupby(TREE_KEY + ['gt_class'])
      .agg(num_views=('correct', 'count'),
           frac_correct_views=('correct', 'mean'))
      .reset_index()
)

print(f"\nTree-level summary: {len(tree_df)} trees")

# Figure : Binned mean accuracy vs. number of views (log-space quantile bins)
log_quantiles = np.quantile(np.log1p(tree_df['num_views']), np.linspace(0, 1, 9))
bins = np.unique(np.expm1(log_quantiles).astype(int))
# bins = np.unique(np.round(np.expm1(log_quantiles)).astype(int))

tree_df['view_bin'] = pd.cut(tree_df['num_views'], bins=bins, include_lowest=True)

bin_stats = tree_df.groupby('view_bin', observed=True).agg(
    mean_acc = ('frac_correct_views', 'mean'),
    sem_acc  = ('frac_correct_views', lambda x: x.std() / np.sqrt(len(x))),
    n_trees  = ('frac_correct_views', 'count'),
).reset_index()

# Geometric midpoint of bin boundaries (midpoint in log space)
bin_stats['bin_mid'] = [np.sqrt(left * right) for left, right in zip(bins[:-1], bins[1:])]

fig, ax = plt.subplots(figsize=(7, 5))
for b in bins:
    ax.axvline(x=b, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.errorbar(bin_stats['bin_mid'], bin_stats['mean_acc'],
            yerr=bin_stats['sem_acc'], fmt='o-', color='steelblue',
            capsize=4, linewidth=2, markersize=7)
for i, (_, row) in enumerate(bin_stats.iterrows()):
    yoffset = 8 if i % 2 == 0 else -16
    ax.annotate(f"n={int(row['n_trees'])}", xy=(row['bin_mid'], row['mean_acc']),
                xytext=(0, yoffset), textcoords='offset points',
                ha='center', fontsize=8, color='gray')
ax.set_xscale('log')
ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=8))
ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs='auto', numticks=20))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.set_xlabel('Number of Views')
ax.set_ylabel('Mean Accuracy +/- SEM')
ax.set_title('Tree-level accuracy vs. number of views per tree', fontweight='bold')
ax.set_ylim(0.58, 1.02)
plt.tight_layout()
out1 = Path("/ofo-share/repos/amritha/tree-species-prediction/evaluation_scripts/figure6_views_vs_accuracy.png")
plt.savefig(out1, bbox_inches='tight')
print(f"Saved: {out1}")
plt.close()