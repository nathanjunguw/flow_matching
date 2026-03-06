import torch
import torch.nn as nn

def flow_loss(model, x_b, x_f, xt, t, t_view, z, dalpha, dbeta, dgamma, train_type='velocity'):
    pred = model(xt, t)
    if train_type == 'velocity':
        I   = dalpha(t_view) * x_b + dbeta(t_view) * x_f + dgamma(t_view) * z
        obj = (0.5 * pred ** 2 - pred * I).mean()
        mse = ((pred - I) ** 2).mean()
    elif train_type == 'denoiser':
        obj = (0.5 * pred ** 2 - pred * z).mean()
        mse = ((pred - z) ** 2).mean()
    else:
        raise ValueError(f"Unknown train_type '{train_type}'. Use 'velocity' or 'denoiser'.")
    return obj, mse, pred