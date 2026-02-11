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
    epoch_selection_interval: int = 2               # how many epochs between recomputing subsets
    use_backbone_gradients: bool = False            # use gradients of backbone as well as classifier head (compute expensive)
    omp_regularizer_strength: float = 0.5           # regularizer to discourage OMP from overfitting (assigning too high of weights to any batch)
    omp_tol: float = 1e-5                           # OMP convergence tolerance (stops when residual ratio < tol)

    # gradient computation method
    use_closed_form_grads: bool = False             # if True, use closed-form gradient for final linear layer (faster, requires linear head)
                                                    # if False, use autograd (more general, works with any head architecture)
    grad_mat_dtype: str = 'float16'                 # dtype for gradient matrix ('float16' or 'float32'); float16 saves VRAM

    normalize_omp_weights: bool = True            # if True, normalize OMP coefficients so mean(nonzero) = 1.0 after selection
    shuffle_before_selection: bool = False          # if True, randomly permute training indices before each selection epoch
                                                    # so batch compositions differ across selection rounds

    # class balancing (for future PerClass selection)
    force_class_balancing: bool = False             # if True, select proportionally from each class (PerClass mode)
    
    
    save_plots: bool = True                         # plots pertain specifically to data reduction / OMP
    save_n_omp_rated_imgs: int = 6                # save the best, worst, and midmost n images as dictated by omp
    simulate_vram: bool = False                 # run a quick VRAM simulation instead of full training;
                                                # allocates tensors at expected sizes without full computation
                                                # to quickly check if training would OOM
dr_config, dr_args = parse_config_args(DataReductionConfig)
