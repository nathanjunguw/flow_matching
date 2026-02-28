import torch
import torch.nn as nn
import math
from torch.optim import AdamW
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt

def sincos_embed(t, time_dimension, base = 10000.0):
    # we cut the number of time dimensions in half
    half = time_dimension // 2
    # we create a tensor of evenly spaced values from 0 to half - 1
    k = torch.arange(half, device = t.device, dtype = t.dtype)
    w_k = base ** (-2 * k / time_dimension)
    # we create a product matrix between the times and w_k
    angles = t[:, None] * w_k[None, :]
    # we compute the sine and cosine and stack them together using torch.cat
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim = 1)


class MLP(nn.Module):
    """
    we have the input size which is the flattened image
    the hidden_sizes are the sizes of the hidden layers
    the t_size it the size we want the time variable to occupy in the initial vector
    we will output a vector where if we put in R^n with time t in R^m, the model
    MLP: R^n times R^m to R^n so it will output back the image
    """
    def __init__(self, input_size, hidden_size, amount_layers, output_size, time_dimension: int):
        super(MLP, self).__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dimension, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.H = hidden_size
        self.L = amount_layers
        self.d_t = time_dimension

        """
        we input the flattened vector which is the image and we want to transform that to the hidden_size.
        """
        self.input_layer = nn.Linear(input_size, hidden_size)

        """
        now we will implement the amount of hidden layers that we have
        """
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(amount_layers)])

        """
        we create a time projection list, where at each layer we will project the time embedded vector e_t tailored
        to that specific layer using nn.Linear(hidden_size, hidden_size). we effectively create a projection for each layer
        so the time will effect each differently
        """
        self.time_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(amount_layers)])

        self.out_layer = nn.Linear(hidden_size, output_size)

        # we elect to use the SiLU function for now
        self.act = nn.SiLU()
    
    def forward(self, x, t):
        phi_t = sincos_embed(t, self.d_t)
        e_t = self.time_mlp(phi_t)

        h = self.input_layer(x)

        for layer, projs in zip(self.hidden_layers, self.time_projs):
            # we add the projection onto it to embed time into the model
            h = h + self.act(layer(h) + projs(e_t))
        
        return self.out_layer(h)


"""
The following pertains to the paths that we can try to use. The list of path types that we can choose from are as follows.

'linear'

'encode_decode'

'variance_preserve'

"""

# loss function for the velocity b(t,x)
def loss_velocity(model, x_b, x_f, xt, t, t_view, z, dalpha, dbeta, dgamma):
    pred = model(xt, t)
    I = dalpha(t_view) * x_b + dbeta(t_view) * x_f + dgamma(t_view) * z

    obj = (0.5 * (pred ** 2) - pred * I).mean()
    mse = ((pred - I) ** 2).mean()
    return obj, mse, pred, I

# loss function for denoiser n_z
def loss_denoiser(model, xt, t, z):
    pred = model(xt, t)

    obj = (0.5 * (pred ** 2) - pred * z).mean()
    mse = ((pred - z) ** 2).mean()
    return obj, mse, pred, z

def gamma_linear(t):
    return torch.zeros_like(t)

def alpha_linear(t):
    return 1.0 - t

def beta_linear(t):
    return t

def dgamma_linear(t):
    return torch.zeros_like(t)

def dalpha_linear(t):
    return -torch.ones_like(t)

def dbeta_linear(t):
    return torch.ones_like(t)

def alpha_variance_preserve(t):
    return torch.cos(math.pi / 2 * t)
def beta_variance_preserve(t):
    return torch.sin(math.pi / 2 * t)
def gamma_variance_preserve(t):
    return torch.zeros_like(t)

def dalpha_variance_preserve(t):
    return (-1 * math.pi / 2) * beta_variance_preserve(t)
def dbeta_variance_preserve(t):
    return (math.pi / 2) * alpha_variance_preserve(t)
def dgamma_variance_preserve(t):
    return torch.zeros_like(t)

c = 0.3  # noise scale

def alpha_trig_noise(t):
    g = c * torch.sin(t * math.pi)
    return torch.cos(t * math.pi / 2) * torch.sqrt(1 - g**2)

def beta_trig_noise(t):
    g = c * torch.sin(t * math.pi)
    return torch.sin(t * math.pi / 2) * torch.sqrt(1 - g**2)

def gamma_trig_noise(t):
    return c * torch.sin(t * math.pi)

def dalpha_trig_noise(t):
    g = c * torch.sin(t * math.pi)
    dg = c * math.pi * torch.cos(t * math.pi)
    a = torch.cos(t * math.pi / 2)
    da = -math.pi/2 * torch.sin(t * math.pi / 2)
    s = torch.sqrt(1 - g**2)
    ds = -g * dg / s
    return da * s + a * ds

def dbeta_trig_noise(t):
    g = c * torch.sin(t * math.pi)
    dg = c * math.pi * torch.cos(t * math.pi)
    b = torch.sin(t * math.pi / 2)
    db = math.pi/2 * torch.cos(t * math.pi / 2)
    s = torch.sqrt(1 - g**2)
    ds = -g * dg / s
    return db * s + b * ds

def dgamma_trig_noise(t):
    return c * math.pi * torch.cos(t * math.pi)

class Interpolant:
    def __init__(self, path):
        self.path = path

        # our alpha/beta/gamma functions are top level functions
        import sys
        self.module = sys.modules[__name__]

        self.alpha = getattr(self.module, f"alpha_{self.path}")
        self.beta = getattr(self.module, f"beta_{self.path}")
        self.gamma = getattr(self.module, f"gamma_{self.path}")
        self.dalpha = getattr(self.module, f"dalpha_{self.path}")
        self.dbeta = getattr(self.module, f"dbeta_{self.path}")
        self.dgamma = getattr(self.module, f"dgamma_{self.path}")
    
    """
    x_base is the base and x_target is the target distribution
    """
    def interpolate(self, x_base, x_target):
        # we get the batch size first
        B = x_base.size(0)
        # we set the device
        device = x_base.device
        # now we get random t values
        t = torch.rand(B, device = device)
        # now we change t to have shape (B, 1)
        t_view = t[:, None]
        # create a gaussian random distribution with same shape as the base
        z = torch.randn_like(x_base)

        x_time = self.alpha(t_view) * x_base + self.beta(t_view) * x_target + self.gamma(t_view) * z
        return x_time, t, t_view, z
    
    def sample_batch(self, BATCH_SIZE, dataset_initial, dataset_target, device):
        N1 = len(dataset_initial["train"])
        N2 = len(dataset_target["train"])

        idx1 = torch.randint(0, N1, (BATCH_SIZE,), device="cpu").tolist()
        idx2 = torch.randint(0, N2, (BATCH_SIZE,), device="cpu").tolist()

        x_b = torch.stack([dataset_initial["train"][i]["pixel_values"] for i in idx1], dim=0).to(device).float()
        x_f = torch.stack([dataset_target["train"][j]["pixel_values"] for j in idx2], dim=0).to(device).float()
        return x_b, x_f
    
    # the two train types that you can put are velocity and denoiser
    def train_model(self, model, dataset_base, dataset_target, train_type = 'velocity', n_iterations = 3000,
                    batch_size = 256, log_every = 200, base_lr = 1e-3, weight_decay = 1e-7,
                    out_name = "MLP_VELOCITY_MODEL.pt"):
        
        optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        loss = getattr(self.module, f"loss_{train_type}")

        for step in range(1, n_iterations+1):
            x_initial, x_target = self.sample_batch(batch_size, dataset_base, dataset_target, device)
            
            xt, t, t_view, z = self.interpolate(x_initial, x_target)

            obj, mse, pred, I = loss(model, x_initial, x_target, xt, t, t_view, z, self.dalpha, self.dbeta, self.dgamma)
            optimizer.zero_grad(set_to_none=True)
            obj.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                with torch.no_grad():
                    pred_mean = pred.mean().item()
                    I_mean = I.mean().item()
                    pred_std = pred.std().item()
                print(f"step {step}/{n_iterations} | obj {obj.item():.6f} | mse {mse.item():.6f} | grad_norm {grad_norm:.4g} | pred_mean {pred_mean:.4g} | I_mean {I_mean:.4g} | pred_std {pred_std:.4f}")

        torch.save(model.state_dict(), out_name)

def new_model(input_size = 4096, hidden_dim=1024, n_layers=4, t_dim=128):
    model = MLP(input_size = input_size, hidden_size=hidden_dim, amount_layers=n_layers, 
                output_size = input_size, time_dimension=t_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

def fetch_model(PATH_NAME, input_size = 4096, hidden_dim=1024, n_layers=4, t_dim=128):
    model = new_model(input_size = input_size, hidden_dim=hidden_dim, n_layers=n_layers, t_dim=t_dim)
    state_dict = torch.load(PATH_NAME, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully from {PATH_NAME}")
    return model

# flattens the dataset so we can pass through the MLP
def flatten_dataset(dataset):
    def flatten_dataset(ex):
        
        x = torch.tensor(ex["pixel_values"])
        ex["pixel_values"] = x.reshape(-1).numpy()
        return ex
    return dataset["train"].map(flatten_dataset)

# we can make the dataset gaussian random
def dataset_to_gaussian(dataset):
    def make_random_gaussian(ex):
        x = torch.tensor(ex["pixel_values"])
        ex["pixel_values"] = torch.randn_like(x).numpy()
        return ex
    return dataset["train"].map(make_random_gaussian)

# function to visualize the flow
def flatten_any(x, img_size=64):
    if x.dim() == 3: return x.reshape(-1)
    if x.dim() == 1: return x
    if x.dim() == 4: return x.reshape(x.size(0), -1)
    if x.dim() == 2: return x
    raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

def unflatten_vec_to_chw(v, C=1, H=64, W=64):
    if v.dim() == 1: return v.view(C, H, W)
    if v.dim() == 2: return v.view(v.size(0), C, H, W)
    raise ValueError(f"Unsupported shape: {tuple(v.shape)}")

@torch.no_grad()
def ode_pushforward_euler_mlp(model, x0, n_steps=800, clamp_x=None, save_frames=9, img_size=64):
    model.eval()
    D = img_size * img_size
    x = flatten_any(x0, img_size=img_size).to(next(model.parameters()).device).float().clone()
    dt = 1.0 / n_steps
    frames, ts = [], []
    save_at = sorted(set(torch.linspace(0, n_steps, steps=save_frames).round().long().tolist()))
    for k in range(n_steps + 1):
        t = k / n_steps
        if k in save_at:
            frames.append(unflatten_vec_to_chw(x.detach().clone(), C=1, H=img_size, W=img_size))
            ts.append(t)
        if k == n_steps: break
        v = model(x.unsqueeze(0), torch.tensor([t], device=x.device).float()).squeeze(0)
        x = x + dt * v
        if clamp_x is not None: x = x.clamp(-clamp_x, clamp_x)
    return unflatten_vec_to_chw(x, C=1, H=img_size, W=img_size), frames, ts

def show_frames(frames, ts, title="ODE trajectory", img_size=64):
    K = len(frames)
    plt.figure(figsize=(3 * K, 3))
    for i, (x, t) in enumerate(zip(frames, ts), 1):
        plt.subplot(1, K, i)
        x_disp = ((x.detach().cpu() + 1.0) / 2.0).clamp(0, 1)
        plt.imshow(x_disp[0], cmap="gray", vmin=0, vmax=1)
        plt.title(f"t={t:.2f}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

def run_flow(dataset_base, model, n_steps=5000, save_frames=9, img_size=64, clamp_x=5.0, use_random=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # pick starting point — random item from dataset or pure noise
    if use_random:
        idx = torch.randint(0, len(dataset_base["train"]), (1,)).item()
        x_src = torch.tensor(dataset_base["train"][idx]["pixel_values"]).view(-1).to(device).float()
        print(f"starting from dataset item {idx}")
    else:
        x_src = torch.randn(img_size * img_size).to(device).float()
        print("starting from random noise")

    x_end, frames, ts = ode_pushforward_euler_mlp(model, x_src, n_steps=n_steps, clamp_x=clamp_x, save_frames=save_frames, img_size=img_size)
    show_frames(frames, ts, img_size=img_size)