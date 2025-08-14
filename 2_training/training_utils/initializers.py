import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import copy

import _bootstrap
from models.TreeSpeciesClassifier import TreeSpeciesClassifierFromPretrained
from data.dataset import TreeDataset, collate_batch
from configs.model_config import model_config
from configs.path_config import path_config
from training_utils.data_utils import stratified_split
from training_utils.image_processing import build_transforms

# when adding weight decay, certain parameters should not be decayed
DECAY_EXCLUDED_PARAM_TYPES = (
    'norm',         # normalization layers
    'embed',        # transformer embedding layers
    'tkn', 'token', # tokenizers
    'bias'          # biases
) # in the below fn, we also exclude all 1D params

def assemble_param_groups(tree_model):
    groups = { # setup param groups for optimizer to control which params get weight decay
        'head_decay': {'params': [], 'names': [], 'lr': model_config.head_lr, 'weight_decay': model_config.head_weight_decay, 'group_name': 'head_decay'},
        'head_nodecay': {'params': [], 'names': [], 'lr': model_config.head_lr, 'weight_decay': 0.0, 'group_name': 'head_nodecay'},
        'backbone_decay':  {'params': [], 'names': [], 'lr': model_config.backbone_lr, 'weight_decay': model_config.backbone_weight_decay, 'group_name': 'backbone_decay'},
        'backbone_nodecay':  {'params': [], 'names': [], 'lr': model_config.backbone_lr, 'weight_decay': 0.0, 'group_name': 'backbone_nodecay'},
    } # names and group names here are not important for the optimizer but kept for reference/debugging

    for name, param in tree_model.named_parameters():
        decay = True
        is_backbone = name.startswith('backbone.')
        if param.ndim < 2 or any(no_decay_type in name.lower() for no_decay_type in DECAY_EXCLUDED_PARAM_TYPES):
            decay = False

        # determine group from above dict keys -> head_ or backbone_ + decay or nodecay
        group_key = ('backbone_' if is_backbone else 'head_') + ('decay' if decay else 'nodecay')
        groups[group_key]['params'].append(param)
        groups[group_key]['names'].append(param)

    return list(groups.values()), groups

def init_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    tree_dset = TreeDataset( # init dataset
        imgs_root=path_config.cropped_tree_training_images / 'labelled',
        gpkg_dir=path_config.drone_crowns_with_field_attributes
    )
    
    # init classifier
    tree_model = TreeSpeciesClassifierFromPretrained(
        path_config.pretrained_model_path, # plant clef 2024, 'only_classifier_then_all' -> fine tuned backbone
        backbone_name="vit_base_patch14_reg4_dinov2.lvd142m",
        num_classes=len(tree_dset.label2idx_map),
        backbone_is_trainable=False, # with already tuned backbone, we won't touch it at least to start (can further tune in later epochs)
    ).to(device)

    # img transforms to give to dataset class to standardize input imgs to the model's liking
    train_transform, val_transform = build_transforms(
        target=model_config.input_img_dim[0],
        long_side_thresh=model_config.downsample_threshold,
        downsample_to=model_config.downsample_to,
        mean=tree_model.backbone_data_cfg['mean'],
        std=tree_model.backbone_data_cfg['std']
    )

    # swap default transform of dataset class with the ones just built
    train_cp = copy.copy(tree_dset)
    train_cp.transform = train_transform
    val_cp = copy.copy(tree_dset)
    val_cp.transform = val_transform

    # train/val split evenly among each label
    train_dset_idxs, val_dset_idxs = stratified_split(tree_dset, per_class_sample_limit_factor=model_config.max_class_imbalance_factor) 
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

    #for name, p in tree_model.named_parameters():
    #    print(f"{name:60s} | shape={tuple(p.shape)} | requires_grad={p.requires_grad}")

    # pass in param groups to optimizer
    # even though backbone is initialized to frozen weights,
    # when unfreezing the params will already in the optimizer
    param_groups, _ = assemble_param_groups(tree_model)
    optim = AdamW(param_groups)
    criterion = nn.CrossEntropyLoss() # loss fn

    # automated mixed precision -> allows for less important calculations with lower fp precision
    enable_amp = model_config.use_amp and device == 'cuda'
    scaler = GradScaler(enabled=enable_amp, device=device) if enable_amp else None

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

    return tree_model, tree_dset, train_loader, val_loader, optim, criterion, scheduler, scaler, device