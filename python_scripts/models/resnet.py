import torch
import torch.nn as nn
import torchvision.models as tvm
from ..utils import sincos_embed

# ══════════════════════════════════════════════════════════════════════════════
#  RESNET
#
#  We use PyTorch's built-in ResNet implementations from torchvision.models.
#  The available variants and their properties are:
#
#    resnet18  : 18 layers,  BasicBlock,      512  feature dim,  ~11M  params
#    resnet34  : 34 layers,  BasicBlock,      512  feature dim,  ~21M  params
#    resnet50  : 50 layers,  BottleneckBlock, 2048 feature dim,  ~25M  params
#    resnet101 : 101 layers, BottleneckBlock, 2048 feature dim,  ~44M  params
#    resnet152 : 152 layers, BottleneckBlock, 2048 feature dim,  ~60M  params
#
#  The original ResNet was designed for ImageNet classification:
#    input  : (B, 3, 224, 224)  -- batch of RGB images
#    output : (B, 1000)         -- scores for 1000 ImageNet categories
#
#  We need to make three modifications to use it for flow matching:
#
#  1. FIX THE INPUT CONV
#     The original conv1 is nn.Conv2d(in_channels=3, ...) hardcoded for RGB.
#     We replace it with nn.Conv2d(in_channels=channels, ...) so it accepts
#     whatever number of channels our images have (e.g. 1 for grayscale).
#     We also change kernel_size from 7 to 3 and stride from 2 to 1, and remove
#     the maxpool, because those were designed for 224x224 ImageNet images and
#     would aggressively throw away spatial information at our smaller image sizes.
#
#  2. REPLACE THE OUTPUT HEAD
#     The original final layer is nn.Linear(512, 1000) which outputs 1000 class
#     scores. We replace it with nn.Identity() to get the raw
#     512 (or 2048) dimensional feature vector out of the backbone, then add our
#     own nn.Linear(feature_dim, input_size) to project to the same size as the
#     input image. We have to read backbone.fc.in_features BEFORE replacing it
#     because nn.Identity() does not have an in_features attribute.
#
#  3. INJECT TIME
#     The original ResNet has no concept of time it just maps images to
#     features. For flow matching we need the model to learn b(t, x), the
#     velocity at time t given image x. We add a small time MLP that embeds
#     the scalar t into a vector of the same size as the feature dim, and add
#     it to the backbone features before the output layer. This way the model
#     can produce different outputs for the same image at different times.
#
#  The full forward pass is:
#    (B, C, H, W)                    -- input image at time t
#        │
#    backbone (modified ResNet)
#        │
#    (B, feature_dim)                -- spatial features collapsed by global avg pool
#        │
#    + time_mlp(sincos_embed(t))     -- add time embedding of same dimension
#        │
#    out linear layer
#        │
#    (B, C*H*W)                      -- predicted velocity, same size as input image
# ══════════════════════════════════════════════════════════════════════════════

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