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

        # build classifier head with proper activations and optional LayerNorm
        self.classifier_head = self._build_classifier_head(
            in_features=self.backbone.num_features,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        print(f"*** Classifier Head Architecture ***")
        for i, layer in enumerate(self.classifier_head):
            print(f"  [{i}] {layer}")

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

    def _build_classifier_head(self, in_features: int, num_classes: int, drop_rate: float) -> nn.Sequential:
        """
        Build classifier head with configurable architecture.

        Architecture options (controlled by model_config):
        - Shallow (n_second_fc_neurons=0): in → [LN] → [Dropout] → FC1 → Act → FC_out
        - Deep (n_second_fc_neurons>0):    in → [LN] → [Dropout] → FC1 → Act → [Dropout] → FC2 → Act → FC_out

        Note: No dropout before final layer (hurts calibration).
        """
        # select activation function
        activation_map = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
        }
        activation_cls = activation_map.get(model_config.head_activation.lower(), nn.GELU)

        layers = []

        # optional LayerNorm to stabilize pretrained features
        if model_config.use_head_layernorm:
            layers.append(nn.LayerNorm(in_features))

        # initial dropout (before first FC)
        if drop_rate > 0:
            layers.append(nn.Dropout(p=drop_rate))

        # first FC layer + activation
        layers.append(nn.Linear(in_features, model_config.n_first_fc_neurons))
        layers.append(activation_cls())

        # optional second FC layer (deep architecture)
        if model_config.n_second_fc_neurons > 0:
            if drop_rate > 0:
                layers.append(nn.Dropout(p=drop_rate))
            layers.append(nn.Linear(model_config.n_first_fc_neurons, model_config.n_second_fc_neurons))
            layers.append(activation_cls())
            final_in_features = model_config.n_second_fc_neurons
        else:
            final_in_features = model_config.n_first_fc_neurons

        # final classification layer (no dropout before this)
        layers.append(nn.Linear(final_in_features, num_classes))

        return nn.Sequential(*layers)

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

    def forward_with_prelogit(self, x):
        """
        Forward pass that also returns the pre-logit features (input to final linear layer).
        Used for closed-form gradient computation in GradMatch.

        Returns:
            logits: (B, num_classes)
            prelogit: (B, final_layer_in_features) - features just before final linear
        """
        feature_tensor = self.backbone(x)  # (B, backbone_feature_dim)

        # pass through all layers except the last one to get pre-logit features
        prelogit = feature_tensor
        for layer in self.classifier_head[:-1]:
            prelogit = layer(prelogit)

        # final linear layer produces logits
        logits = self.classifier_head[-1](prelogit)

        return logits, prelogit

    def get_final_layer_dims(self):
        """Returns (in_features, out_features) of the final linear layer."""
        final_layer = self.classifier_head[-1]
        return final_layer.in_features, final_layer.out_features

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