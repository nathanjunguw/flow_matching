
import torch
from sklearn.datasets import make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt

def get_gaussian_matrix(rows, cols, mu, sigma):
    """
    Generates a Gaussian matrix of shape (rows, cols) with mean mu and standard deviation sigma.
    """
    return torch.normal(mean=mu, std=sigma, size=(rows, cols))

def get_swiss_roll(n_samples, noise=0.0):
    """
    Generates a Swiss Roll dataset with n_samples and optional noise.
    """
    X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    # take only the x and z columns
    X = X[:, [0,2]]
    # standardize the dataset for viewing
    X = (X - X.mean(0)) / X.std(0)
    return torch.tensor(X, dtype=torch.float32)

def plot_swiss_roll_2D(swiss_roll, image_size):
    """
    Will give the swiss roll in the square image size of image_size by image_size
    Input: 2D swiss roll with shape (N, 2)
    """
    # we convert to numpy first
    X = swiss_roll.detach().cpu().numpy()
    # subtract the minimum value of each column from the column
    X_n = X - X.min(0)
    X_n = X_n / X_n.max(0)
    X_n *= (image_size - 1)

    img = np.zeros((image_size, image_size), dtype = np.float32)
    xi = X_n[:, 0].astype(int).clip(0, image_size - 1)
    yi = X_n[:, 1].astype(int).clip(0, image_size - 1)
    np.add.at(img, (xi, yi), 1)

    img /= img.max()
    img = 1 - img * 2

    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap="gray", vmin = -1, vmax = 1)
    plt.axis('off')
    plt.show()

    return torch.tensor(img, dtype=torch.float32)

def see_grayscale(data, image_size):
    """
    we will show the grayscale in -1 to 1
    input: must be a tensor and for this instance it must be flattened
    """
    img = np.zeros((image_size, image_size), dtype = np.float32)
    X = data.reshape(image_size, image_size)
    X.detach().cpu().numpy()

    plt.figure(figsize=(4,4))
    plt.imshow(X, cmap="gray", vmin = -1, vmax = 1)
    plt.axis('off')
    plt.show()



