import torch
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional, Literal

from utils.config_utils import parse_config_args, register_yaml_str_representers

@dataclass
class TreeModelConfig:
    """
    Config class to control hyperparams for model training

    Note: the `input_img_dim` is dictated by what the plantclef model was trained on,
        furthermore, any img dims should ALWAYS be multiples of 14,
        as the dinov2 arch plantclef uses learns 14x14 patches.
    
    'head' refers to the classification head we are training from scratch
        This is the final linear layer that receives feature vectors from the pretrained model,
        since we removed this layer from the pretrained model since it was made for different classes/labels

    'backbone' refers to the pretrained model we are using/tuning

    A mix of papers, the original plantclef model args, youtube videos, and GPT dictated cfg values here,
    they will likely need experimenting/tuning.
    However it is known that the backbone needs smaller adjustments than the head.
    """
    # input data/data preprocessing
    input_img_dim: tuple[int, int] = (518, 518) # size to scale input imgs to before giving to model
    downsample_threshold: int = 574             # if longer edge of input img > this, downsample instead of just crop
    downsample_to: int = 518                    # size to downsample long edge too before padding
    num_workers: int = 8                        # workers for the dataloader
    max_class_imbalance_factor: float = 0      # 0 -> no limiting factor; if class A has n samples, class B has m samples, 
                                                # will subsample class A to be at most `max_class_imbalance_factor` * m samples
    min_samples_per_class: int = 500            # 0 -> no limit; exclude classes with fewer than this num samples
    
    # epoch loop iterations
    epochs: int = 25                            # num passes through the training dataset
    warmup_epochs: int = 2                      # how many epochs spent slowly incr lr
    freeze_backbone_epochs: int = 2             # keep backbone frozen for first N epochs
    batch_size: int = 16                         # how many images processed per backprop/param update
    
    # parameter stepping
    head_lr: float = 1e-3                       # learning rate: how big of a step to take down gradient when updating model params
    backbone_lr: float = 1.0e-4     
    head_weight_decay: float = 1e-2             # factor to regularize weights towards 0
    backbone_weight_decay: float = 5e-3         
    
    # model architecture
    n_intermediate_fc_layer_neurons: int = 1024 # size of fc layer between input of backbone and output logits of classification head
                                                # set to 0 for no intermediate layer

    # optimizations
    n_last_layers_to_unfreeze: int = 8          # unfreezing all layers causes OOM errors, choose how many of the last layers to make tunable
    layer_unfreeze_step: int = 2                # each epoch how many layers to unfreeze UP TO `n_last_layers_to_unfreeze`
    use_amp: bool = True                        # automated mixed precision
    amp_dtype: torch.dtype = torch.float16      # dtype for amp scaling
    use_data_reduction: bool = False            # gradient based subset selection (see `configs/data_reduction_config.py``)
    
    # early stopping
    patience: int = 0                           # how many consecutive epochs must be same or worse before stopping training early
                                                # enter 0 to disable early stopping
    improvement_margin: float = 0.              # how much better does prev epoch have to be to not stop early
    # metric tracked to determine when to stop (corresponds to metrics dict returned by `step_epoch()`)
    monitor_metric: Literal['f1', 'recall', 'precision', 'accuracy', 'loss'] = 'f1'
    objective: Literal ['max', 'min'] = 'max'   # dictates to stopper whether 'best' means higher or lower

    # misc
    seed: int = 24                              # seed to maintain reproducibility within rng
    ckpt_dir_tag: str = ''                      # ckpt dirs are just date and time, use this for a more helpful training identifier


model_config, model_args = parse_config_args(TreeModelConfig)
register_yaml_str_representers(torch.dtype) # YAML dump doesn't natively know what to do with torch types