import torch
import torch.nn as nn
from torchvision import transforms as T
import timm
from pathlib import Path

from configs.model_config import model_config

class TreeSpeciesClassifierFromPretrained(nn.Module):
    """
    Re-uses the PlantCLEF ViT-DINOv2 backbone, drops its 7,806-way classifier head,
    and attaches a fresh classifier head for tree species set.

    Parameters:

    checkpoint_path : str | Path
        Path to PlantCLEF checkpoint
    num_classes     : int
        Number of species in dataset
    backbone_name   : str
        timm model name that matches the checkpoint
    drop_rate       : float, default 0.1
        Dropout before the new classification layer
    freeze_backbone : bool, default False
        If True, backbone weights are frozen (only the new head trains).
    """
    def __init__(
        self,
        ckpt_path: Path | str,   # path to pretrained classifier ckpt
        backbone_name: str,     # what *kind* of model our pretrained ckpt is (see timm docs for supported models)
        num_classes: int,
        backbone_is_trainable: bool,  # false -> only train our new classifier head; False -> tune pretrained weights and class head
        drop_rate: float = 0.1,
    ):
        super().__init__()

        # init the base ViT architecture that was used for pretrained model
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,   # load weights manually
            num_classes=0,      # removes original classifier head (so we attach ours)
            global_pool='avg'   # output a flat feature vector from this base model
        )

        self.backbone_data_cfg = timm.data.resolve_model_data_config(self.backbone)
        
        # load weights from pretrained
        # stay on cpu for now to avoid fragmentation and allow for easier modifications in init
        ckpt_path = Path(ckpt_path)
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']

        # strict=False here allows for us to init the weights without the original classification head
        self.backbone.load_state_dict(state_dict, strict=False) 

        # un/freeze backbone
        self.toggle_backbone_weights_trainability(backbone_is_trainable)

        # append a new classification head with 2 FC layers and optional dropout
        layers = []
        if drop_rate > 0:
            layers.append(nn.Dropout(p=drop_rate))
        layers.append(nn.Linear(self.backbone.num_features, model_config.n_intermediate_fc_layer_neurons))
        layers.append(nn.Linear(model_config.n_intermediate_fc_layer_neurons, num_classes))
        self.classifier_head = nn.Sequential(*layers)

        # transforms similar to DINOv2 pre normalization
        backbone_cfg = timm.data.resolve_model_data_config(self.backbone.pretrained_cfg)
        size = backbone_cfg["input_size"][1:]    # (224, 224) patches for ViT-B/14
        self.train_transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=backbone_cfg["mean"], std=backbone_cfg["std"]),
        ])

        self.eval_transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=backbone_cfg["mean"], std=backbone_cfg["std"]),
        ])

    def head_parameters(self):
        return [p for p in self.classifier_head.parameters() if p.requires_grad]

    def toggle_backbone_weights_trainability(self, backbone_is_trainable):
        for p in self.backbone.parameters():
            p.requires_grad = backbone_is_trainable

    def unfreeze_last_n_backbone_layers(self, n):
        backbone_blocks = self.backbone.blocks
        total_blocks = len(backbone_blocks)
        n_unfreeze = min(n, total_blocks)
        for i in range(total_blocks - n_unfreeze, total_blocks):
            for p in backbone_blocks[i].parameters():
                p.requires_grad = True
        
        print(f"*** {n_unfreeze} Backbone layers unfrozen. Target num unfrozen layers: {model_config.n_last_layers_to_unfreeze} --- Total Backbone layers: {total_blocks}")

    def forward(self, x):
        feature_tensor = self.backbone(x) # (B, back_bone_feature_dim)
        return self.classifier_head(feature_tensor) # (B, num_classes)

class TreeMetaDataMLP(nn.Module):
    '''
    placeholder to later include some metadata about the tree with the img itself for classifying
    this can include drone height, tree angle relative to camera, nadir/oblique etc.
    '''
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embed_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()