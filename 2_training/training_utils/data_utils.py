import torch
from torch.utils.data import Subset
import geopandas as gpd
from pathlib import Path

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

def stratified_split(dset, val_ratio=0.2, per_class_sample_limit_factor=None, seed=-1):
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
    if per_class_sample_limit_factor is not None:
        sample_limit = int(min(counts) * per_class_sample_limit_factor)
    else:
        sample_limit = None

    rng = torch.Generator().manual_seed(seed)
    train_idxs, val_idxs = [], []
    for c in classes:
        idxs = torch.nonzero(labels == c).squeeze(1) # get all samples of class c
        n_samples_in_class = idxs.numel()

        if per_class_sample_limit_factor is not None:
            n_samples = min(sample_limit, n_samples_in_class)
        else:
            n_samples = n_samples_in_class
        
        perm = torch.randperm(n_samples, generator=rng) # random permutation of numbers
        idxs = idxs[perm]

        n_val_samples = int(n_samples_in_class * val_ratio)
        val_idxs.extend(idxs[:n_val_samples].tolist())
        train_idxs.extend(idxs[n_val_samples:].tolist())

    # final extra shuffle of all the idxs
    train_shuffle_idxs = torch.randperm(len(train_idxs), generator=rng).tolist()
    val_shuffle_idxs = torch.randperm(len(val_idxs), generator=rng).tolist()
    train_idxs = torch.tensor(train_idxs)[train_shuffle_idxs]
    val_idxs = torch.tensor(val_idxs)[val_shuffle_idxs]

    return train_idxs, val_idxs