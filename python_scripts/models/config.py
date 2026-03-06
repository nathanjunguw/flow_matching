import torch.nn as nn
from dataclasses import dataclass
from .mlp import MLP, MLP_Residual
from .resnet import ResNet
from .unet import UNet


@dataclass
class FlowModelConfig:
    """
    Describes which model architecture to use and all of its hyperparameters.

    model_type     : 'mlp' | 'mlp_residual' | 'resnet' | 'unet'

    Shared

    img_size       : spatial side-length of the square input image
    channels       : image channels (1=grayscale, 3=RGB)
    t_dim          : sinusoidal time-embedding dimension (not used by UNet)

    MLP / MLP_Residual

    hidden_dim     : width of hidden layers
    n_layers       : number of hidden layers

    ResNet

    hidden_dim     : feature width
    resnet_variant : 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152'

    UNet
    
    unet_variant   : 'unet_small' | 'unet_base' | 'unet_large' | 'unet_xlarge'
    """

    model_type     : str   = 'mlp'
    img_size       : int   = 64
    channels       : int   = 1
    t_dim          : int   = 128
    # MLP / MLP_Residual
    hidden_dim     : int   = 1024
    n_layers       : int   = 4
    # ResNet
    resnet_variant : str   = 'resnet34'
    # UNet
    unet_variant   : str   = 'unet_base'

    @property
    def needs_flatten(self) -> bool:
        """Only MLP variants need flat (C*H*W,) input. ResNet and UNet take (C, H, W)."""
        return self.model_type in ('mlp', 'mlp_residual')

    @property
    def flat_size(self) -> int:
        return self.channels * self.img_size * self.img_size

    def build(self) -> nn.Module:
        """Instantiate and return the model on CPU."""
        if self.model_type == 'mlp':
            return MLP(self.flat_size, self.hidden_dim, self.n_layers, self.t_dim)
        elif self.model_type == 'mlp_residual':
            return MLP_Residual(self.flat_size, self.hidden_dim, self.n_layers, self.flat_size, self.t_dim)
        elif self.model_type == 'resnet':
            return ResNet(self.flat_size, self.channels, self.resnet_variant, self.t_dim)
        elif self.model_type == 'unet':
            return UNet(self.channels, self.unet_variant)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. "
                             f"Choose 'mlp', 'mlp_residual', 'resnet', or 'unet'.")

    def save(self, model: nn.Module, path: str):
        import torch
        torch.save({'config': self, 'state_dict': model.state_dict()}, path)
        print(f"Saved to {path}")

    @staticmethod
    def load(path: str):
        """Returns (FlowModelConfig, nn.Module) with weights loaded."""
        import torch
        ckpt  = torch.load(path, map_location='cpu')
        cfg   = ckpt['config']
        model = cfg.build()
        model.load_state_dict(ckpt['state_dict'])
        print(f"Loaded from {path}")
        return cfg, model

def list_options():
    print("model_type:")
    print("  'mlp'          -- plain feedforward, no skip connections")
    print("  'mlp_residual' -- feedforward with residual skip connections")
    print("  'resnet'       -- torchvision ResNet, takes (B, C, H, W)")
    print("  'unet'         -- diffusers UNet2DModel, takes (B, C, H, W)")
    print()
    print("resnet_variant:")
    print("  'resnet18'     -- 18 layers,  BasicBlock,      512  feature dim, ~11M params")
    print("  'resnet34'     -- 34 layers,  BasicBlock,      512  feature dim, ~21M params")
    print("  'resnet50'     -- 50 layers,  BottleneckBlock, 2048 feature dim, ~25M params")
    print("  'resnet101'    -- 101 layers, BottleneckBlock, 2048 feature dim, ~44M params")
    print("  'resnet152'    -- 152 layers, BottleneckBlock, 2048 feature dim, ~60M params")
    print()
    print("unet_variant:")
    print("  'unet_small'   -- (32, 64, 128),           1 block/level, lightest")
    print("  'unet_base'    -- (64, 128, 256, 512),     2 blocks/level, good default")
    print("  'unet_large'   -- (128, 256, 512, 1024),   2 blocks/level, more capacity")
    print("  'unet_xlarge'  -- (256, 512, 1024, 1024),  3 blocks/level, heaviest")
    print()
    print("interpolant path:")
    print("  'linear'             -- straight interpolation, gamma=0, ODE only")
    print("  'variance_preserve'  -- cosine schedule,        gamma=0, ODE only")
    print("  'trig_noise'         -- cosine + noise,         gamma>0, ODE or SDE")
    print()
    print("stepping:")
    print("  'ode' -- Euler integration,            one model  (velocity)")
    print("  'sde' -- Euler-Maruyama integration,   two models (velocity + denoiser)")