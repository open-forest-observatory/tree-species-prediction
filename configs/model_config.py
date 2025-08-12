import torch
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

from utils.config_utils import parse_config_args

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
    input_img_dim: tuple[int, int] = (518, 518) # size to scale input imgs to before giving to model
    downsample_threshold: int = 574             # if longer edge of input img > this, downsample instead of just crop
    downsample_to: int = 518                    # size to downsample long edge too before padding
    epochs: int = 20                            # num passes through the training dataset
    batch_size: int = 128                       # how many images processed per backprop/param update
    head_lr: float = 1e-3                       # learning rate: how big of a step to take down gradient when updating model params
    backbone_lr: float = 1.0e-4     
    head_weight_decay: float = 1e-2             # factor to regularize weights towards 0
    backbone_weight_decay: float = 5e-3         
    warmup_epochs: int = 2                      # how many epochs spent slowly incr lr
    freeze_backbone_epochs: int = 2             # keep backbone frozen for first N epochs
    num_workers: int = 0                        # workers for the dataloader
    use_amp: bool = True                        # automated mixed precision
    amp_dtype: torch.dtype = torch.float16      # dtype for amp scaling
    seed: int = 24                              # seed to maintain reproducibility within rng


model_config, model_args = parse_config_args(TreeModelConfig)