import math
import sys
import torch
from datasets import DatasetDict

#  INTERPOLANT PATHS
#  Each path name 'foo' must have functions:
#    alpha_foo, beta_foo, gamma_foo, dalpha_foo, dbeta_foo, dgamma_foo

# ── linear  (gamma = 0)

def alpha_linear(t):    return 1.0 - t
def beta_linear(t):     return t
def gamma_linear(t):    return torch.zeros_like(t)
def dalpha_linear(t):   return -torch.ones_like(t)
def dbeta_linear(t):    return torch.ones_like(t)
def dgamma_linear(t):   return torch.zeros_like(t)

# ── variance_preserve  (gamma = 0)

def alpha_variance_preserve(t):  return torch.cos(math.pi / 2 * t)
def beta_variance_preserve(t):   return torch.sin(math.pi / 2 * t)
def gamma_variance_preserve(t):  return torch.zeros_like(t)
def dalpha_variance_preserve(t): return (-math.pi / 2) * torch.sin(math.pi / 2 * t)
def dbeta_variance_preserve(t):  return ( math.pi / 2) * torch.cos(math.pi / 2 * t)
def dgamma_variance_preserve(t): return torch.zeros_like(t)

# ── trig_noise  (gamma = c*sin(pi*t), nonzero -> SDE stepping is meaningful)

_NOISE_C = 0.3

def _g(t):    return _NOISE_C * torch.sin(t * math.pi)
def _dg(t):   return _NOISE_C * math.pi * torch.cos(t * math.pi)
def _s(t):    return torch.sqrt(1 - _g(t) ** 2)
def _ds(t):   return -_g(t) * _dg(t) / _s(t)

def alpha_trig_noise(t):   return torch.cos(t * math.pi / 2) * _s(t)
def beta_trig_noise(t):    return torch.sin(t * math.pi / 2) * _s(t)
def gamma_trig_noise(t):   return _g(t)

def dalpha_trig_noise(t):
    a  = torch.cos(t * math.pi / 2);  da = -math.pi / 2 * torch.sin(t * math.pi / 2)
    return da * _s(t) + a * _ds(t)

def dbeta_trig_noise(t):
    b  = torch.sin(t * math.pi / 2);  db = math.pi / 2 * torch.cos(t * math.pi / 2)
    return db * _s(t) + b * _ds(t)

def dgamma_trig_noise(t):  return _dg(t)

class Interpolant:
    """
    Stochastic interpolant  X_t = alpha(t)*x_base + beta(t)*x_target + gamma(t)*z

    Parameters
    ----------
    path : str
        'linear', 'variance_preserve', or 'trig_noise'.
        Paths with gamma != 0 ('trig_noise') support SDE stepping.
    """

    def __init__(self, path: str):
        self.path = path
        mod = sys.modules[__name__]
        for fn in ('alpha', 'beta', 'gamma', 'dalpha', 'dbeta', 'dgamma'):
            setattr(self, fn, getattr(mod, f"{fn}_{path}"))

    @property
    def has_noise(self) -> bool:
        """True when gamma is not identically zero (SDE stepping is meaningful)."""
        return self.gamma(torch.tensor([0.5])).abs().item() > 1e-8

    def interpolate(self, x_base: torch.Tensor, x_target: torch.Tensor):
        """
        Sample t ~ Uniform[0,1] and compute X_t.
        Returns: x_time, t (B,), t_view (broadcastable), z (noise)
        """
        B      = x_base.size(0)
        device = x_base.device
        t      = torch.rand(B, device=device)
        # t_view broadcasts over all non-batch dims (works for flat and spatial)
        t_view = t.view(B, *([1] * (x_base.dim() - 1)))
        z      = torch.randn_like(x_base)
        x_time = self.alpha(t_view) * x_base + self.beta(t_view) * x_target + self.gamma(t_view) * z
        return x_time, t, t_view, z

    def sample_batch(self, batch_size: int, ds_base: DatasetDict,
                     ds_target: DatasetDict, device: torch.device):
        """Draw a random mini-batch from each dataset."""
        N1   = len(ds_base["train"])
        N2   = len(ds_target["train"])
        idx1 = torch.randint(0, N1, (batch_size,)).tolist()
        idx2 = torch.randint(0, N2, (batch_size,)).tolist()
        x_b  = torch.stack([ds_base["train"][i]["pixel_values"]   for i in idx1]).to(device).float()
        x_f  = torch.stack([ds_target["train"][j]["pixel_values"] for j in idx2]).to(device).float()
        return x_b, x_f