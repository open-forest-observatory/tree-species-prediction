import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.config_utils import parse_config_args


@dataclass
class TreeModelConfig:
    # placeholder config class for model training parameters

    num_classes: int = 5
    in_height: int = 1024
    in_width: int = 1024


model_config, model_args = parse_config_args(TreeModelConfig)
