import torch
import torch.nn as nn
from typing import Optional
from datasets import DatasetDict
from torch.optim import AdamW
from tqdm import tqdm

from .models import FlowModelConfig
from .interpolants import Interpolant
from .loss import flow_loss
from .utils import (
    _ode_euler, _sde_euler_maruyama, _show_frames,
    _flow_batch, _dataset_to_gaussian, filter_dataset,
)


class FlowExperiment:
    """
    The main entry point. Bundles model config + interpolant + stepping mode.

    Parameters
    ----------
    config      : FlowModelConfig
    interpolant : Interpolant
    stepping    : 'ode' or 'sde'
        'ode' -- train one velocity model, integrate with Euler.
        'sde' -- train a velocity model AND a denoiser, integrate with
                 Euler-Maruyama. Only meaningful when interpolant.has_noise=True.
    """

    def __init__(self, config: FlowModelConfig, interpolant: Interpolant,
                 stepping: str = 'ode'):
        self.config      = config
        self.interpolant = interpolant
        self.stepping    = stepping
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if stepping == 'sde' and not interpolant.has_noise:
            print(f"[Warning] stepping='sde' but interpolant '{interpolant.path}' has gamma=0. "
                  "SDE and ODE are identical for this path.")

        self.model_velocity: nn.Module          = config.build().to(self.device)
        self.model_denoiser: Optional[nn.Module] = None
        if stepping == 'sde':
            self.model_denoiser = config.build().to(self.device)

    #  TRAIN

    def train(
        self,
        dataset_base:   DatasetDict,
        dataset_target: DatasetDict,
        n_iterations:   int   = 3000,
        batch_size:     int   = 256,
        base_lr:        float = 1e-3,
        weight_decay:   float = 1e-7,
        log_every:      int   = 200,
        rand_run:       bool  = False,
        out_name:       str   = 'flow_experiment',
    ) -> dict:
        """
        Train the experiment's model(s).

        Parameters
        rand_run  : replace x_base with fresh Gaussian noise each step
                    (unconditional generation from pure noise).
        out_name  : base filename for checkpoints.
                    Saves <out_name>_velocity.pt  and  <out_name>_denoiser.pt

        Returns
        dict
            'velocity_loss' : list[float]  -- per-step objective values
            'denoiser_loss' : list[float]  -- per-step values, or [] if ODE-only

        The loss lists can be passed directly to plot_loss() whenever you like.
        """
        kw = dict(n_iterations=n_iterations, batch_size=batch_size, base_lr=base_lr,
                  weight_decay=weight_decay, log_every=log_every, rand_run=rand_run)

        print("── Training velocity model ──")
        vel_loss = self._train_one(
            self.model_velocity, dataset_base, dataset_target,
            train_type='velocity', out_path=f"{out_name}_velocity.pt", **kw
        )

        den_loss = []
        if self.stepping == 'sde' and self.model_denoiser is not None:
            print("\n── Training denoiser model ──")
            den_loss = self._train_one(
                self.model_denoiser, dataset_base, dataset_target,
                train_type='denoiser', out_path=f"{out_name}_denoiser.pt", **kw
            )

        return {'velocity_loss': vel_loss, 'denoiser_loss': den_loss}

    # ─────────────────────────────────────────────────────────────────────────────
    #  _train_one
    #
    #  Core training loop for a single model (either velocity or denoiser).
    #  Called twice by train() when stepping='sde', once when stepping='ode'.
    #
    #  Before the loop starts, one forward pass is run with no_grad to print
    #  the initial objective and mse so you have a baseline to compare against
    #  as training progresses.
    #
    #  During the loop, tqdm shows a progress bar that updates in place with
    #  the current obj, mse, and grad_norm every log_every steps. This is
    #  cleaner than printing a new line every log_every steps.
    #
    #  Returns a list of per-step objective values (one per gradient step)
    #  which can be passed to plot_loss() whenever you want to visualise
    #  the training curve.
    # ─────────────────────────────────────────────────────────────────────────────

    def _train_one(self, model, dataset_base, dataset_target, train_type,
               n_iterations, batch_size, base_lr, weight_decay, log_every,
               rand_run, out_path) -> list:
    
        interp    = self.interpolant
        optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations)
        model.train()
        history = []

        # compute initial stats before any training
        with torch.no_grad():
            x_b, x_f = interp.sample_batch(batch_size, dataset_base, dataset_target, self.device)
            if rand_run:
                x_b = torch.randn_like(x_b)
            xt, t, t_view, z      = interp.interpolate(x_b, x_f)
            obj0, mse0, pred0     = flow_loss(model, x_b, x_f, xt, t, t_view, z,
                                            interp.dalpha, interp.dbeta, interp.dgamma, train_type)
        print(f"initial | obj {obj0.item():.4f} | mse {mse0.item():.4f} | "
            f"pred_mean {pred0.mean().item():.4f} | pred_std {pred0.std().item():.4f}")

        pbar = tqdm(range(1, n_iterations + 1), desc=f"Training {train_type}")

        for step in pbar:
            x_b, x_f = interp.sample_batch(batch_size, dataset_base, dataset_target, self.device)
            if rand_run:
                x_b = torch.randn_like(x_b)

            xt, t, t_view, z = interp.interpolate(x_b, x_f)
            obj, mse, pred   = flow_loss(model, x_b, x_f, xt, t, t_view, z,
                                        interp.dalpha, interp.dbeta, interp.dgamma, train_type)
            history.append(obj.item())

            optimizer.zero_grad(set_to_none=True)
            obj.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
                percent   = allocated / total * 100
                pbar.set_postfix({
                    'obj':  f"{obj.item():.4f}",
                    'mse':  f"{mse.item():.4f}",
                    'grad': f"{grad_norm:.3f}",
                    'memory': f"{allocated:.2f} GB / {total:.2f} GB ({percent:.2f}%)"
                })

        self.config.save(model, out_path)
        return history

    #  VISUALISE

    def visualise(
        self,
        dataset_base:     DatasetDict,
        n_steps:          int   = 1000,
        save_frames:      int   = 9,
        clamp_x:          Optional[float] = 5.0,
        use_random_start: bool  = True,
        use_gaussian_start: bool = True,
        # SDE-specific
        eps_max:   float = 0.05,
        eps_power: float = 2.0,
        gamma_max: float = 1.0,
        eps_floor: float = 0.0,
    ):
        """
        Push one starting sample through the flow and display the trajectory.
        Uses ODE or SDE stepping according to self.stepping.
        """
        x_src  = self._pick_start(dataset_base, use_random_start)
        if use_gaussian_start:
            x_src = torch.randn_like(x_src)
        kwargs = dict(n_steps=n_steps, clamp_x=clamp_x, save_frames=save_frames,
                      img_size=self.config.img_size, channels=self.config.channels,
                      needs_flatten=self.config.needs_flatten)

        if self.stepping == 'ode' or self.model_denoiser is None:
            _, frames, ts = _ode_euler(self.model_velocity, x_src, **kwargs)
            title = f"ODE  |  {self.config.model_type}  |  {self.interpolant.path}"
        else:
            _, frames, ts = _sde_euler_maruyama(
                self.model_velocity, self.model_denoiser, x_src,
                gamma=self.interpolant.gamma,
                eps_max=eps_max, eps_power=eps_power,
                gamma_max=gamma_max, eps_floor=eps_floor,
                **kwargs,
            )
            title = f"SDE  |  {self.config.model_type}  |  {self.interpolant.path}"

        _show_frames(frames, ts, title=title)

    #  FID

    def fid(self, dataset_target: DatasetDict, steps: int = 1000,
            batch_size: int = 64) -> float:
        """
        Pixel-space Frechet distance between generated samples and the real dataset.
        Flows Gaussian noise through the velocity ODE, filters outliers, then
        compares mean and covariance with real data.
        """
        from scipy.linalg import sqrtm

        gaussian_ds = DatasetDict({"train": _dataset_to_gaussian(dataset_target)})
        generated   = _flow_batch(
            self.model_velocity, gaussian_ds,
            n_steps=steps, device=self.device, batch_size=batch_size,
            img_size=self.config.img_size, channels=self.config.channels,
            needs_flatten=self.config.needs_flatten,
        )
        generated = filter_dataset(generated)

        real = np.array(dataset_target["train"]["pixel_values"]).reshape(len(dataset_target["train"]), -1)
        fake = np.array(generated["train"]["pixel_values"]).reshape(len(generated["train"]), -1)

        mu_r, mu_f   = real.mean(0), fake.mean(0)
        sig_r, sig_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
        diff         = mu_r - mu_f
        eps          = 1e-6
        I            = np.eye(sig_r.shape[0])
        covmean      = sqrtm((sig_r + eps * I) @ (sig_f + eps * I)).real
        fid_val      = float(diff @ diff + np.trace(sig_r + sig_f - 2 * covmean))
        print(f"FID: {fid_val:.4f}")
        return fid_val

    #  LOAD WEIGHTS

    def load_weights(self, out_name: str = 'flow_experiment'):
        """Reload saved checkpoints into the experiment's model(s)."""
        _, vel_state = torch.load(f"{out_name}_velocity.pt", map_location='cpu').values()
        self.model_velocity.load_state_dict(vel_state)
        if self.stepping == 'sde' and self.model_denoiser is not None:
            _, den_state = torch.load(f"{out_name}_denoiser.pt", map_location='cpu').values()
            self.model_denoiser.load_state_dict(den_state)
        print("Weights loaded.")

    # ─────────────────────────────────────────────────────────────────────────

    def _pick_start(self, dataset_base: DatasetDict, use_random: bool) -> torch.Tensor:
        if use_random:
            idx   = torch.randint(0, len(dataset_base["train"]), (1,)).item()
            x_src = torch.tensor(dataset_base["train"][idx]["pixel_values"]).to(self.device).float()
            print(f"Starting from dataset item {idx}")
        else:
            if self.config.needs_flatten:
                x_src = torch.randn(self.config.flat_size, device=self.device)
            else:
                x_src = torch.randn(self.config.channels, self.config.img_size,
                                    self.config.img_size, device=self.device)
            print("Starting from random Gaussian noise")
        return x_src