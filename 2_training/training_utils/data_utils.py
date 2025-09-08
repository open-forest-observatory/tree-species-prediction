import torch
from torch.utils.data import Subset, DataLoader
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import copy

from configs.path_config import path_config
from configs.model_config import model_config
from data.dataset import collate_batch

def get_classes_from_gpd_file_paths(paths):
    """
    Take in a Path object or list of Path objects that point to gpkg files
    Get the species_codes from all and return the unique ones

    Essentially finds all unique tree species across multiple gpkg files
    """
    if not isinstance(paths, list) and isinstance(paths, (Path, str)):
        paths = list(paths) # if given single path and not a list, wrap as a list

    species = set()
    for dset_path in paths:
        dset_path = Path(dset_path)
        gdf = gpd.read_file(dset_path)
        gdf_species = gdf[gdf['species_code'].notnull()].species_code
        species.update(list(gdf_species.unique()))

    return sorted(species)

def stratified_split(dset, val_ratio=0.2, per_class_sample_limit_factor=0, min_samples_per_class=0, seed=-1):
    """
    Returns (train_subset, val_subset) with class-balanced split.

    Parameters:
    dataset: TreeDataset (must expose .meta with 'label_idx')
    val_ratio: float   fraction of samples that go to validation
    seed: int     random seed for reproducibility
    """
    labels = torch.tensor([m['label_idx'] for m in dset.meta], dtype=torch.long)
    classes, counts = torch.unique(labels, return_counts=True)
    classes = classes.tolist()

    # determine max samples per class if given `per_class_sample_limit_factor`
    if per_class_sample_limit_factor > 0:
        sample_limit = int(min(counts) * per_class_sample_limit_factor)
    else:
        sample_limit = None

    rng = torch.Generator().manual_seed(seed)
    train_idxs, val_idxs = [], []
    for c in classes:
        idxs = torch.nonzero(labels == c).squeeze(1) # get all samples of class c
        n_samples_in_class = idxs.numel()

        if min_samples_per_class > 0 and n_samples_in_class < min_samples_per_class:
            continue

        if per_class_sample_limit_factor > 0:
            n_samples = min(sample_limit, n_samples_in_class)
        else:
            n_samples = n_samples_in_class
        
        perm = torch.randperm(n_samples, generator=rng) # random permutation of numbers
        idxs = idxs[perm]

        n_val_samples = int(n_samples * val_ratio)
        val_idxs.extend(idxs[:n_val_samples].tolist())
        train_idxs.extend(idxs[n_val_samples:].tolist())

    # final extra shuffle of all the idxs
    train_shuffle_idxs = torch.randperm(len(train_idxs), generator=rng).tolist()
    val_shuffle_idxs = torch.randperm(len(val_idxs), generator=rng).tolist()
    train_idxs = torch.tensor(train_idxs)[train_shuffle_idxs]
    val_idxs = torch.tensor(val_idxs)[val_shuffle_idxs]

    return train_idxs, val_idxs

def stratified_split_by_ID(
    dset,
    val_ratio=0.2,
    per_class_sample_limit_factor=0,
    min_samples_per_class=0,
    sample_idxs_pool=None,
    seed=-1,
    suppress_summary_print=False,
):
    """
    Returns (train_subset, val_subset) with class-balanced split while keeping
    all images from the same unique_treeID together in the same split.

    Assumptions:
      - Each unique_treeID belongs to exactly one class (label_idx).
      - dset.meta[i] has keys: 'label_idx' (int) and 'unique_treeID' (hashable/str).

    Strategy:
      - Work per-class, but group samples by unique_treeID.
      - For each class: pick whole-tree groups for val so that the *total images*
        in val are as close as possible to `val_ratio * n_samples_in_class`.
      - Remaining trees go to train. Final shuffle keeps RNG reproducible.
    """
    rng = torch.Generator().manual_seed(seed)
    pool = list(range(len(dset))) if (sample_idxs_pool is None) else list(sample_idxs_pool)

    if sample_idxs_pool is None:
        pool = list(range(len(dset)))
    else:
        pool = list(sample_idxs_pool)

    # tensors for labels and a list of unique_treeIDs
    labels = torch.tensor([dset.meta[i]['label_idx'] for i in pool], dtype=torch.long)
    tree_ids = [dset.meta[i]['unique_treeID'] for i in pool]

    tree_to_label = {}
    tree_to_indices = defaultdict(list)

    classes, counts = torch.unique(labels, return_counts=True)
    classes = classes.tolist()

    # optional per-class cap on #samples (images)
    sample_limit = None
    if per_class_sample_limit_factor > 0:
        sample_limit = int(int(counts.min()) * per_class_sample_limit_factor)

    # build groups: per unique_treeID -> (label, idx_list)
    # also verify consistency: a tree shouldn't have mixed labels
    tree_to_label = {}
    tree_to_indices = defaultdict(list)

    for global_idx, (tid, lbl) in zip(pool, zip(tree_ids, labels.tolist())):
        tree_to_indices[tid].append(global_idx)  # store global idx
        if tid in tree_to_label and tree_to_label[tid] != lbl:
            raise ValueError(f"unique_treeID {tid} has mixed labels: {tree_to_label[tid]} vs {lbl}")
        tree_to_label[tid] = lbl

    # invert: per-class list of (unique_treeID, size, indices)
    class_groups = {c: [] for c in classes}
    for tid, lbl in tree_to_label.items():
        idxs = tree_to_indices[tid]
        class_groups[lbl].append((tid, len(idxs), idxs))

    train_idxs, val_idxs = [], []

    for c in classes:
        groups = class_groups[c]
        if not groups:
            continue

        # total images in this class
        total_images_c = sum(sz for _, sz, _ in groups)
        if min_samples_per_class > 0 and total_images_c < min_samples_per_class:
            # skip undersized class entirely
            continue

        # apply optional per-class cap (by images) by taking whole trees until ~cap
        if sample_limit is not None and total_images_c > sample_limit:
            # shuffle groups, then greedily add while staying <= cap if possible
            order = torch.randperm(len(groups), generator=rng).tolist()
            capped_groups, running = [], 0
            for j in order:
                tid, sz, idxs = groups[j]
                if running + sz <= sample_limit:
                    capped_groups.append((tid, sz, idxs))
                    running += sz
            # if we couldn't add anything (rare huge tree), just take the smallest tree
            if not capped_groups:
                smallest = min(groups, key=lambda g: g[1])
                capped_groups = [smallest]
                running = smallest[1]
            groups = capped_groups
            total_images_c = running

        # target #images for validation in this class
        target_val = int(round(total_images_c * val_ratio))

        if target_val == 0 and len(groups) > 1:
            # still keep at least one small tree in val to avoid 0
            target_val = 1

        # pick whole trees for val close to target_val (greedy + backoff)
        # shuffle groups
        perm = torch.randperm(len(groups), generator=rng).tolist()
        groups = [groups[j] for j in perm]

        val_set, val_sum = [], 0
        for tid, sz, idxs in groups:
            if val_sum + sz <= target_val:
                val_set.append((tid, sz, idxs))
                val_sum += sz

        # if we undershot and adding the next group overshoots but is closer, allow it
        if val_sum < target_val:
            remaining = [g for g in groups if g[0] not in {t for t, _, _ in val_set}]
            # choose the single group that gets us closest to target (even if it overshoots)
            best = None
            best_diff = None
            for tid, sz, idxs in remaining:
                diff = abs((val_sum + sz) - target_val)
                if best is None or diff < best_diff:
                    best = (tid, sz, idxs)
                    best_diff = diff
            if best is not None:
                # only add if it actually improves closeness
                if best_diff < abs(val_sum - target_val):
                    val_set.append(best)
                    val_sum += best[1]

        # everything not in val_set goes to train
        val_tree_ids = {t for t, _, _ in val_set}
        for tid, sz, idxs in groups:
            (val_idxs if tid in val_tree_ids else train_idxs).extend(idxs)

    # final reproducible shuffle
    if train_idxs:
        train_idxs = torch.tensor(train_idxs, dtype=torch.long)
        train_idxs = train_idxs[torch.randperm(len(train_idxs), generator=rng)]
    else:
        train_idxs = torch.tensor([], dtype=torch.long)

    if val_idxs:
        val_idxs = torch.tensor(val_idxs, dtype=torch.long)
        val_idxs = val_idxs[torch.randperm(len(val_idxs), generator=rng)]
    else:
        val_idxs = torch.tensor([], dtype=torch.long)

    if sample_idxs_pool is not None:
        assigned = set(train_idxs.tolist()) | set(val_idxs.tolist())
        assert assigned == set(sample_idxs_pool), "Some pooled samples were not assigned"

    # summary of train/val splits after stratified split and applying any min/max n_samples
    if not suppress_summary_print:
        summarize_split_by_tree(dset, train_idxs, name="train")
        summarize_split_by_tree(dset, val_idxs,   name="val")
        check_no_tree_overlap(dset, train_idxs, val_idxs)

    return train_idxs, val_idxs

# TODO: implement the class balancing factors here that are in the other stratified split fns
def stratified_split_by_plot(dset, per_class_sample_limit_factor=None, min_samples_per_class=None, suppress_summary_print=False):
    split_df = pd.read_csv(path_config.train_test_split_file)
    target_cols = ['plot_id', 'mission_id_hn', 'mission_id_lo'] # these cols contain the info to construct the indv gpkg file name

    train_rows = split_df[split_df.split == 'train']
    train_dset_names = train_rows.apply(lambda row: f"{int(row[target_cols[0]]):04d}_{int(row[target_cols[1]]):06d}_{int(row[target_cols[2]]):06d}", axis=1).tolist()

    test_rows = split_df[split_df.split == 'test']
    test_dset_names = test_rows.apply(lambda row: f"{int(row[target_cols[0]]):04d}_{int(row[target_cols[1]]):06d}_{int(row[target_cols[2]]):06d}", axis=1).tolist()

    train_idxs, test_idxs, unknown = [], [], []
    for i, meta_dict in enumerate(dset.meta):
        img_src = meta_dict['dset']
        if img_src in test_dset_names:
            test_idxs.append(i)
        elif img_src in train_dset_names:
            train_idxs.append(i)
        
        else:
            unknown.append(i)

    print(f"****** {len(train_idxs)} - {len(test_idxs)}")

    # ensure class coverage between plots
    train_labels = {dset.meta[i]['label_idx'] for i in train_idxs}
    test_labels = {dset.meta[i]['label_idx'] for i in test_idxs}

    labels_only_in_train = sorted(train_labels - test_labels)
    labels_only_in_test = sorted(test_labels - train_labels)
    
    print(f"[split_by_dataset_id] train imgs={len(train_idxs)}, test imgs={len(test_idxs)}, unknown dsets={len(unknown)}")
    if labels_only_in_train:
        print(f"Classes ONLY in TRAIN (absent in TEST): {labels_only_in_train}")
    if labels_only_in_test:
        print(f"Classes ONLY in TEST (absent in TRAIN): {labels_only_in_test}")
    if unknown:
        ex = sorted(set(unknown))[:5]
        print(f"Warning: {len(unknown)} samples had dset names not found in the split CSV. Examples: {ex}")

    # summary of train/val splits after stratified split and applying any min/max n_samples
    if not suppress_summary_print:
        summarize_split_by_tree(dset, train_idxs, name="train")
        summarize_split_by_tree(dset, test_idxs,   name="val")
        check_no_tree_overlap(dset, train_idxs, test_idxs)

    return torch.tensor(train_idxs), torch.tensor(test_idxs)

def summarize_split_by_tree(dset, idxs, name="split"):
    """
    Print, per class, the number of trees and images, plus per-tree count stats.
    Assumes dset.meta[i] has 'label_idx' and 'unique_treeID'.
    """
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.cpu().numpy()
    else:
        idxs = np.asarray(idxs, dtype=int)

    labels = np.array([dset.meta[i]['label_idx']     for i in idxs])
    trees  = np.array([dset.meta[i]['unique_treeID'] for i in idxs])

    classes = np.unique(labels)
    total_imgs = len(idxs)
    total_trees = np.unique(trees).size
    print(f"\n=== {name.upper()} SUMMARY ===")
    print(f"total: {total_imgs} images across {total_trees} trees")

    for c in classes:
        mask = (labels == c)
        trees_c = trees[mask]
        uniq_trees, counts = np.unique(trees_c, return_counts=True)
        n_trees = uniq_trees.size
        n_imgs  = int(counts.sum())
        if n_trees == 0:
            continue
        print(
            f"Class {c}: {n_trees:4d} trees, {n_imgs:5d} images | "
            f"per-tree imgs → min {counts.min():2d}, "
            f"median {int(np.median(counts)):2d}, mean {counts.mean():.2f}, max {counts.max():2d}"
        )

def check_no_tree_overlap(dset, train_idxs, val_idxs):
    """Assert no unique_treeID appears in both splits."""
    to_np = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=int)
    train_idxs = to_np(train_idxs); val_idxs = to_np(val_idxs)
    train_trees = {dset.meta[i]['unique_treeID'] for i in train_idxs}
    val_trees   = {dset.meta[i]['unique_treeID'] for i in val_idxs}
    overlap = train_trees & val_trees
    if overlap:
        print(f"WARNING: {len(overlap)} trees appear in both splits (showing up to 10): {list(sorted(overlap))[:10]}")
    else:
        print("OK: no tree overlap between TRAIN and VAL.")

def assemble_dataloaders(tree_dset, train_transform, val_transform, split_method, return_idxs=False, idxs_pool=None):
    train_cp = copy.copy(tree_dset)
    val_cp = copy.copy(tree_dset)

    split_methods = { # choose appropriate fn for splitting data based on model config arg
        'plot': stratified_split_by_plot,
        'tree': stratified_split_by_ID,
        'image': stratified_split
    }

    # train/val split evenly among each label
    train_dset_idxs, val_dset_idxs = split_methods[split_method](
        tree_dset,
        per_class_sample_limit_factor=model_config.max_class_imbalance_factor,
        min_samples_per_class=model_config.min_samples_per_class
    ) 

    # swap default transform of dataset class with the ones just built
    train_cp.transform = train_transform
    val_cp.transform = val_transform

    train_dset = Subset(train_cp, train_dset_idxs)
    val_dset = Subset(val_cp, val_dset_idxs)

    train_loader = DataLoader(
        train_dset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=model_config.num_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )

    if return_idxs:
        return train_loader, val_loader, train_dset_idxs, val_dset_idxs
    else:
        return train_loader, val_loader