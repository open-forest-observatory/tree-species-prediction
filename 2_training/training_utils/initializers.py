import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import copy

import _bootstrap
from models.TreeSpeciesClassifier import TreeSpeciesClassifierFromPretrained
from data.dataset import TreeDataset
from configs.model_config import model_config
from configs.path_config import path_config
from utils.config_utils import kwargs_from_config
from training_utils.data_utils import assemble_dataloaders
from training_utils.early_stop import EarlyStopper
from training_utils.image_processing import build_transforms

# when adding weight decay, certain parameters should not be decayed
DECAY_EXCLUDED_PARAM_TYPES = (
    'norm',         # normalization layers
    'embed',        # transformer embedding layers
    'tkn', 'token', # tokenizers
    'bias'          # biases
) # in the below fn, we also exclude all 1D params

def assemble_param_groups(tree_model, n_backbone_layers_to_include: int = 0):
    """
    Build parameter groups for optimizer with separate lr/decay for head vs backbone.

    Args:
        tree_model: the model
        n_backbone_layers_to_include: number of backbone layers (from the end) to include in optimizer.
            If 0, no backbone params are added (head-only training).
            If > 0, only the last N blocks are included (for gradual unfreezing).
    """
    groups = {
        'head_decay': {'params': [], 'names': [], 'lr': model_config.head_lr, 'weight_decay': model_config.head_weight_decay, 'group_name': 'head_decay'},
        'head_nodecay': {'params': [], 'names': [], 'lr': model_config.head_lr, 'weight_decay': 0.0, 'group_name': 'head_nodecay'},
    }

    # only create backbone groups if we'll actually tune backbone layers
    if n_backbone_layers_to_include > 0:
        groups['backbone_decay'] = {'params': [], 'names': [], 'lr': model_config.backbone_lr, 'weight_decay': model_config.backbone_weight_decay, 'group_name': 'backbone_decay'}
        groups['backbone_nodecay'] = {'params': [], 'names': [], 'lr': model_config.backbone_lr, 'weight_decay': 0.0, 'group_name': 'backbone_nodecay'}

    # determine which backbone block indices will be unfrozen
    backbone_blocks = tree_model.backbone.blocks
    total_blocks = len(backbone_blocks)
    unfreeze_start_idx = total_blocks - n_backbone_layers_to_include

    # debug: print decay exclusion check for embeddings/tokens
    print("*** Checking weight decay exclusions for embeddings/tokens ***")
    for name, param in tree_model.named_parameters():
        if any(t in name.lower() for t in ['embed', 'token', 'tkn', 'cls']):
            excluded = param.ndim < 2 or any(no_decay_type in name.lower() for no_decay_type in DECAY_EXCLUDED_PARAM_TYPES)
            print(f"  {name}: ndim={param.ndim}, excluded_from_decay={excluded}")

    for name, param in tree_model.named_parameters():
        is_backbone = name.startswith('backbone.')

        # skip backbone params that won't be unfrozen
        if is_backbone and n_backbone_layers_to_include > 0:
            # check if this param belongs to a block that will be unfrozen
            # format: backbone.blocks.{idx}.xxx
            if '.blocks.' in name:
                try:
                    block_idx = int(name.split('.blocks.')[1].split('.')[0])
                    if block_idx < unfreeze_start_idx:
                        continue  # skip frozen blocks
                except (ValueError, IndexError):
                    continue  # skip if can't parse
            elif not any(x in name for x in ['norm', 'head']):
                # skip non-block backbone params (patch_embed, cls_token, etc.) unless norm/head
                continue
        elif is_backbone and n_backbone_layers_to_include == 0:
            continue  # skip all backbone params

        # determine decay
        decay = True
        if param.ndim < 2 or any(no_decay_type in name.lower() for no_decay_type in DECAY_EXCLUDED_PARAM_TYPES):
            decay = False

        # determine group from above dict keys -> head_ or backbone_ + decay or nodecay
        group_key = ('backbone_' if is_backbone else 'head_') + ('decay' if decay else 'nodecay')
        groups[group_key]['params'].append(param)
        groups[group_key]['names'].append(name)

    # filter out empty groups
    param_groups = [g for g in groups.values() if len(g['params']) > 0]

    group_summary = [f"{g['group_name']} ({len(g['params'])} params)" for g in param_groups]
    print(f"*** Parameter groups: {group_summary} ***")

    return param_groups, groups

def init_training():
    tree_dset = TreeDataset( # init dataset
        imgs_root=path_config.cropped_tree_training_images / 'labelled',
        gpkg_dir=path_config.drone_crowns_with_field_attributes,
        cache_dir=path_config.static_transformed_images_cache_dir,
    )
    
    # init classifier
    tree_model = TreeSpeciesClassifierFromPretrained(
        path_config.pretrained_model_path, # plant clef 2024, 'only_classifier_then_all' -> fine tuned backbone
        backbone_name=model_config.backbone_name,
        num_classes=len(tree_dset.label2idx_map),
        drop_rate=model_config.drop_rate,
        backbone_is_trainable=False, # with already tuned backbone, we won't touch it at least to start (can further tune in later epochs)
    ).to(model_config.device)

    # img transforms to give to dataset class to standardize input imgs to the model's liking
    static_T, random_train_T, random_val_T = build_transforms(
        target=model_config.input_img_dim[0],
        long_side_thresh=model_config.downsample_threshold,
        downsample_to=model_config.downsample_to,
        mean=tree_model.backbone_data_cfg['mean'],
        std=tree_model.backbone_data_cfg['std']
    )

    train_loader, val_loader, train_subset, val_subset = assemble_dataloaders(
        tree_dset,
        static_T,
        random_train_T,
        random_val_T,
        model_config.data_split_level,
        upper_limit_n_samples=model_config.max_total_samples,
        return_subsets=True,
        idxs_pool=None,
        plot_sample_imgs=False,
        val_ratio=model_config.val_ratio
    )

    # controls early stopping of training if performance plateaus
    # disabled if model_config.patience == 0
    early_stopper = EarlyStopper(**kwargs_from_config(model_config, EarlyStopper))
    early_stopper.lag_epochs = (model_config.n_last_layers_to_unfreeze * model_config.layer_unfreeze_step) + model_config.freeze_backbone_epochs

    #for name, p in tree_model.named_parameters():
    #    print(f"{name:60s} | shape={tuple(p.shape)} | requires_grad={p.requires_grad}")

    # pass in param groups to optimizer
    # only include backbone params for layers that will eventually be unfrozen
    # if n_last_layers_to_unfreeze == 0, no backbone params are added (head-only)
    param_groups, _ = assemble_param_groups(
        tree_model,
        n_backbone_layers_to_include=model_config.n_last_layers_to_unfreeze
    )
    optim = AdamW(param_groups)

    # loss function with optional label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=model_config.label_smoothing)

    # automated mixed precision -> allows for less important calculations with lower fp precision
    enable_amp = model_config.use_amp and model_config.device == 'cuda'
    scaler = GradScaler(enabled=enable_amp, device=model_config.device) if enable_amp else None

    #for name, p in tree_model.named_parameters():
    #    print(f"{name:60s} | shape={tuple(p.shape)} | requires_grad={p.requires_grad}")

    # schedulers -> warmup and anneal
    if model_config.warmup_epochs > 0:
        assert model_config.epochs > model_config.warmup_epochs # cannot train for a total num epochs < the warmup
        
        # warmup slowly increases lr linearly over the course of `warmup_epochs` -> helps with fine tuning ViTs
        warmup = LinearLR(optim, start_factor=1e-3, total_iters=model_config.warmup_epochs)
        
        # after `warmup_epochs` we will decay lr following a cosine curve towards 0 -> allows finer changes towards end of training
        annealer = CosineAnnealingLR(optim, T_max=model_config.epochs - model_config.warmup_epochs)
        scheduler = SequentialLR(optim, schedulers=[warmup, annealer], milestones=[model_config.warmup_epochs])
    else:
        # use just cosine annealing if no warmup
        scheduler = CosineAnnealingLR(optim, T_max=model_config.epochs)

    return tree_model, tree_dset, train_loader, val_loader, train_subset, val_subset, static_T, \
    random_train_T, random_val_T, optim, criterion, scheduler, scaler, early_stopper