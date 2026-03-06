import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from tqdm import tqdm

def sincos_embed(t: torch.Tensor, time_dimension: int, base: float = 10000.0) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar time values, shape (B,) -> (B, time_dimension)."""
    half   = time_dimension // 2
    k      = torch.arange(half, device=t.device, dtype=t.dtype)
    w_k    = base ** (-2 * k / time_dimension)
    angles = t[:, None] * w_k[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

def prime_dataset(dataset: DatasetDict, model_type: str, img_size: int = 64) -> DatasetDict:
    """
    Prepare a raw image DatasetDict for training.

    MLP / ResNet -> pixel_values flattened to a 1-D vector  (C*H*W,)
    UNet -> pixel_values kept as (C, H, W) tensor
    """
    needs_flat = model_type in ('mlp', 'mlp_residual')

    def _process(ex):
        x = torch.tensor(ex["pixel_values"])
        if x.dim() == 2:        # (H, W) -> add channel dim
            x = x.unsqueeze(0)
        if needs_flat:
            x = x.reshape(-1)
        ex["pixel_values"] = x.numpy()
        return ex

    dataset["train"] = dataset["train"].map(_process)
    dataset["train"].set_format("torch", columns=["pixel_values"])
    return dataset


def filter_dataset(dataset: DatasetDict, min_val: float = -2.0, max_val: float = 2.0,
                   threshold: float = 0.9) -> DatasetDict:
    """Remove samples where fewer than `threshold` fraction of values are in [min_val, max_val]."""
    samples = np.array(dataset["train"]["pixel_values"])
    flat    = samples.reshape(len(samples), -1)
    mask    = np.array([((s >= min_val) & (s <= max_val)).mean() >= threshold
                        for s in tqdm(flat, desc="Filtering")])
    print(f"Kept {mask.sum()}/{len(mask)} samples  (threshold={threshold})")
    return DatasetDict({"train": Dataset.from_dict({"pixel_values": samples[mask]})})


def _dataset_to_gaussian(dataset: DatasetDict) -> Dataset:
    """Replace every pixel_values entry with i.i.d. Gaussian noise of the same shape."""
    def _rand(ex):
        x = torch.tensor(ex["pixel_values"])
        ex["pixel_values"] = torch.randn_like(x).numpy()
        return ex
    return dataset["train"].map(_rand)

#  LOW-LEVEL INTEGRATORS

def _prep_x(x: torch.Tensor, device, needs_flatten: bool,
            channels: int, img_size: int) -> torch.Tensor:
    """Ensure x is on the right device and has the right shape for the model."""
    x = x.to(device).float()
    if needs_flatten:
        return x.reshape(-1)
    return x.view(channels, img_size, img_size) if x.dim() == 1 else x


def _to_chw(x: torch.Tensor, channels: int, img_size: int, needs_flatten: bool) -> torch.Tensor:
    """Return a (C, H, W) clone for display, regardless of current shape."""
    v = x.detach().clone()
    return v.view(channels, img_size, img_size) if needs_flatten else v


@torch.no_grad()
def _ode_euler(model, x0, n_steps=800, clamp_x=None, save_frames=9,
               img_size=64, channels=1, needs_flatten=True):
    """Euler integration of  dX/dt = b(t, X)."""
    model.eval()
    device = next(model.parameters()).device
    x      = _prep_x(x0, device, needs_flatten, channels, img_size)
    dt     = 1.0 / n_steps
    frames, ts = [], []
    save_at = set(torch.linspace(0, n_steps, steps=save_frames).round().long().tolist())

    for k in range(n_steps + 1):
        t = k / n_steps
        if k in save_at:
            frames.append(_to_chw(x, channels, img_size, needs_flatten))
            ts.append(t)
        if k == n_steps:
            break
        t_t = torch.tensor([t], device=device, dtype=torch.float32)
        v   = model(x.unsqueeze(0), t_t).squeeze(0)
        x   = x + dt * v
        if clamp_x is not None:
            x = x.clamp(-clamp_x, clamp_x)

    return _to_chw(x, channels, img_size, needs_flatten), frames, ts


@torch.no_grad()
def _sde_euler_maruyama(model_velocity, model_denoiser, x0, gamma,
                        n_steps=800, clamp_x=None, save_frames=9,
                        img_size=64, channels=1, needs_flatten=True,
                        eps_max=0.05, eps_power=2.0, gamma_max=1.0, eps_floor=0.0):
    """
    eps(t) = eps_max * (|gamma(t)| / gamma_max)^eps_power + eps_floor
    This ensures eps -> 0 at the endpoints where gamma -> 0.
    """
    model_velocity.eval()
    model_denoiser.eval()
    device  = next(model_velocity.parameters()).device
    x       = _prep_x(x0, device, needs_flatten, channels, img_size)
    dt      = 1.0 / n_steps
    sqrt_dt = math.sqrt(dt)
    frames, ts = [], []
    save_at = set(torch.linspace(0, n_steps, steps=save_frames).round().long().tolist())

    for k in range(n_steps + 1):
        t = k / n_steps
        if k in save_at:
            frames.append(_to_chw(x, channels, img_size, needs_flatten))
            ts.append(t)
        if k == n_steps:
            break

        t_t    = torch.tensor([t], device=device, dtype=torch.float32)
        g      = gamma(t_t).abs()
        g_norm = (g / gamma_max).clamp(0.0, 1.0)
        eps_t  = (eps_max * g_norm ** eps_power + eps_floor).clamp_min(0.0)

        b     = model_velocity(x.unsqueeze(0), t_t).squeeze(0)
        eta   = model_denoiser(x.unsqueeze(0), t_t).squeeze(0)
        ratio = eps_t / g.clamp_min(1e-12)

        x = x + dt * (b - ratio * eta) + torch.sqrt(2.0 * eps_t) * sqrt_dt * torch.randn_like(x)
        if clamp_x is not None:
            x = x.clamp(-clamp_x, clamp_x)

    return _to_chw(x, channels, img_size, needs_flatten), frames, ts


def _show_frames(frames, ts, title="Flow trajectory"):
    K = len(frames)
    plt.figure(figsize=(3 * K, 3))
    for i, (f, t) in enumerate(zip(frames, ts), 1):
        plt.subplot(1, K, i)
        img = ((f.cpu() + 1.0) / 2.0).clamp(0, 1)
        if img.shape[0] == 1:
            plt.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        else:
            plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"t={t:.2f}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def _flow_batch(model, dataset, n_steps=100, device=None, batch_size=64,
                img_size=64, channels=1, needs_flatten=True):
    """Flow every sample in a dataset through the ODE, batch by batch."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    raw     = np.array(dataset["train"]["pixel_values"])
    images  = torch.tensor(raw, dtype=torch.float32)
    dt      = 1.0 / n_steps
    outputs = []
    n_blown = 0

    for i in tqdm(range(0, len(images), batch_size), desc="Flowing dataset"):
        x     = images[i:i + batch_size].to(device)
        B     = x.size(0)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        for step in range(n_steps):
            t     = torch.full((B,), step * dt, device=device)
            v     = model(x, t)
            x     = x + v * dt
            blown = torch.isnan(x).view(B, -1).any(1) | torch.isinf(x).view(B, -1).any(1)
            alive = alive & ~blown
        valid    = x[alive]
        n_blown += (~alive).sum().item()
        if len(valid):
            outputs.append(valid.cpu())

    print(f"Flowing done. Removed {n_blown} samples due to blowup.")
    out = torch.cat(outputs, 0).numpy()
    return DatasetDict({"train": Dataset.from_dict({"pixel_values": out})})

def plot_loss(loss_history: list, title: str = "Training objective", window: int = 50):
    import pandas as pd
    threshold = loss_history[0]
    clipped   = [v if v <= threshold else None for v in loss_history]
    smoothed  = pd.Series(clipped).rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed)
    plt.xlabel("Gradient step")
    plt.ylabel("Objective")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_size_and_channel(dataset):
    sample = torch.tensor(dataset['train']['pixel_values'][0])

    # get the image size (assuming the image is square)
    IMAGE_SIZE = sample.shape[1]
    CHANNELS = 1 if sample.dim() == 2 else sample.shape[0]
    return IMAGE_SIZE, CHANNELS