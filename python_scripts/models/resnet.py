import torch
import torch.nn as nn
import torchvision.models as tvm
from ..utils import sincos_embed

import torchvision.models as tvm

_RESNET_CONSTRUCTORS = {
    'resnet18':  tvm.resnet18,
    'resnet34':  tvm.resnet34,
    'resnet50':  tvm.resnet50,
    'resnet101': tvm.resnet101,
    'resnet152': tvm.resnet152,
}

class ResNet(nn.Module):
    def __init__(self, input_size: int, channels: int, variant: str, t_dim: int):
        super().__init__()
        if variant not in _RESNET_CONSTRUCTORS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_RESNET_CONSTRUCTORS)}")
        
        backbone = _RESNET_CONSTRUCTORS[variant](weights=None)
        
        # fix input conv for channel count, and patch for small images
        # originally the resnet woudl expect channels to be 3 for RGB images but
        # we have the instance that we could be putting in grayscale images
        backbone.conv1  = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        
        # pull out feature dim before replacing the head
        feature_dim = backbone.fc.in_features   # 512 for 18/34, 2048 for 50/101/152
        # backbone.fc = nn.Linear(512, 1000), but we don't want it to output 1000 we want it to keep
        # the 512 and then we want to pass it through to output our input size
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.d_t      = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim),
        )

        # we go from the feature_dim to the input_size to output something of the same size
        self.out = nn.Linear(feature_dim, input_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # have to implement the time embedding
        e_t   = self.time_mlp(sincos_embed(t, self.d_t))
        return self.out(feats + e_t)