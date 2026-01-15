from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

from utils.config_utils import parse_config_args

@dataclass
class DataReductionConfig:
    seed: Optional[int] = None

    strategy: Optional[str] = 'gradmatch'           # options are ['gradmatch', 'random']
    subset_ratio: Optional[float] = 0.5
    num_warm_start_epochs: int = 0                  # number of epochs to train on the full dataset before subsetting
    #subbatch_size: int = 32                         # set to 1 for per sample selection (set as low as possible before OMP takes too long)
    epoch_selection_interval: int = 5               # how many epochs between recomputing subsets
    use_backbone_gradients: bool = False            # use gradients of backbone as well as classifier head (compute expensive)
    omp_regularizer_strength: float = 0.5           # regularizer to discourage OMP from overfitting (assigning too high of weights to any batch)
    omp_tol: float = 1e-5                          # regularizer to discourage OMP from overfitting (assigning too high of weights to any batch)
    
    
    save_plots: bool = True                         # plots pertain specifically to data reduction / OMP
    save_n_omp_rated_imgs: int = 10                # save the best, worst, and midmost n images as dictated by omp

dr_config, dr_args = parse_config_args(DataReductionConfig)
