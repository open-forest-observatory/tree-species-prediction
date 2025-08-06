import torch
import torch.nn as nn
import timm

class TreeSpeciesClassifierFromPretrained(nn.module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        use_metadata: bool = False,
        meta_input_dim: int = 2,           # e.g. sinθ, cosθ
        meta_hidden_dims: list[int] = (32, 32),
        meta_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        

class TreeMetaDataMLP(nn.module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embed_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()