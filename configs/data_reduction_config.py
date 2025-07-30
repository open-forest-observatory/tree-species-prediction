from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

from utils.config_utils import parse_config_args

@dataclass
class DataReductionConfig:
    seed: Optional[int] = None

    analyze: bool = False # will run future analysis scripts for previous experiments rather than running experiments
    data_reduction_method: Optional[str] = None
    subset_ratio: Optional[float] = None
    num_warm_start_epochs: int = 100 # number of epochs to train on the full dataset before subsetting
    subbatch_size: int = 250 # set to 1 for per sample selection (compute expensive)
    epoch_selection_interval: int = 50 # percent of total epochs to recompute subsets
    patience: int = 100
    use_jit: bool = False

dr_config, dr_args = parse_config_args(DataReductionConfig)
