from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

from utils.config_utils import parse_config_args

@dataclass
class DataReductionConfig:
    seed: Optional[int] = None

    data_reduction_method: Optional[str] = None
    subset_ratio: Optional[float] = 0.5
    num_warm_start_epochs: int = 0                  # number of epochs to train on the full dataset before subsetting
    subbatch_size: int = 1                          # set to 1 for per sample selection (set as low as possible before OMP takes too long)
    epoch_selection_interval: int = 1               # how many epochs between recomputing subsets
    use_backbone_gradients: bool = False            # use gradients of backbone as well as classifier head (compute expensive)

dr_config, dr_args = parse_config_args(DataReductionConfig)
