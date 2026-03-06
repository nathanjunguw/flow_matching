import torch
import torch.nn as nn
#  UNET
#
#  We use the UNet2DModel from Hugging Face's diffusers library rather than
#  writing our own. It is a standard 2D UNet designed specifically for
#  generative modelling with time conditioning already built in.
#
#  Input  : (B, C, H, W); batch of images, no flattening needed
#  Output : (B, C, H, W); same shape as input, predicted velocity field
#
#  The variants differ in how many channels each level has and how many
#  conv blocks per level, controlling the capacity of the network:
#
#    unet_small  : (32, 64, 128),          1 block/level
#    unet_base   : (64, 128, 256, 512),    2 blocks/level
#    unet_large  : (128, 256, 512, 1024),  2 blocks/level
#    unet_xlarge : (256, 512, 1024, 1024), 3 blocks/level
#
#  CONSTRAINT: img_size must be divisible by 2^(num levels)

_UNET_CONFIGS = {
    'unet_small':  dict(
        block_out_channels = (32, 64, 128),
        layers_per_block   = 1,
        down_block_types   = ('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'),
        up_block_types     = ('AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    ),
    'unet_base':   dict(
        block_out_channels = (64, 128, 256, 512),
        layers_per_block   = 2,
        down_block_types   = ('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
        up_block_types     = ('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    ),
    'unet_large':  dict(
        block_out_channels = (128, 256, 512, 1024),
        layers_per_block   = 2,
        down_block_types   = ('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
        up_block_types     = ('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    ),
    'unet_xlarge': dict(
        block_out_channels = (256, 512, 1024, 1024),
        layers_per_block   = 3,
        down_block_types   = ('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
        up_block_types     = ('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    ),
}

class UNet(nn.Module):
    def __init__(self, channels: int, variant: str):
        super().__init__()
        from diffusers import UNet2DModel
        if variant not in _UNET_CONFIGS:
            raise ValueError(f"Unknown UNet variant '{variant}'. "
                             f"Choose from: {list(_UNET_CONFIGS.keys())}")
        self.model = UNet2DModel(
            in_channels  = channels,
            out_channels = channels,
            **_UNET_CONFIGS[variant],
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t).sample