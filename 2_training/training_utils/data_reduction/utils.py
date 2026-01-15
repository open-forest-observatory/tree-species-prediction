import torch
from torch.utils.data import DataLoader, Subset
import copy

import _bootstrap
from training_utils.data_reduction.omp import OrthogonalMP_REG_Parallel_V1
from data.dataset import collate_batch
from configs.data_reduction_config import dr_config
from configs.model_config import model_config

class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, y, meta = self.subset[int(i)]
        meta = dict(meta)              # avoid mutating shared
        meta["subset_idx"] = int(i)    # always int
        return img, y, meta

def make_selection_loader(tree_dset, train_subset, static_T, val_T):
    # Copy base dataset so transforms don't interfere with the training dataset object
    sel_cp = copy.copy(tree_dset)
    sel_cp.static_transform = static_T
    sel_cp.random_transform = val_T

    sel_subset = Subset(sel_cp, train_subset.indices)
    sel_loader = DataLoader(
        sel_subset,
        batch_size=model_config.batch_size,
        shuffle=False, # deterministic for selection
        num_workers=model_config.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return sel_loader

def rebuild_train_loader(train_subset, chosen_subset_indices):
    chosen_train_subset = Subset(train_subset, chosen_subset_indices)
    train_loader = DataLoader(
        IndexedSubset(chosen_train_subset),
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=model_config.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return train_loader