import torch
import torch.nn as nn
from ..utils import sincos_embed

#  MLP  (plain feedforward, no skip connections)
#
#  The simplest possible neural network for flow matching. Takes a flattened
#  image and a time value and outputs a vector of the same size.
#
#  Each hidden layer does:
#    h = SiLU( Linear(h) + time_proj(e_t) )
#
#  The forward pass is:
#    (B, C*H*W) flattened image
#    input_layer: Linear(input_size, hidden_size)
#    for each hidden layer:
#        SiLU( Linear(h) + time_proj(e_t) )
#    out_layer: Linear(hidden_size, input_size)
#    (B, C*H*W)

class MLP(nn.Module):
    """
    Plain feedforward network with NO skip connections.

    Each hidden layer is:  Linear → SiLU,  with the time embedding
    added as a bias shift before the activation.  There is no residual path —
    each layer fully overwrites h.

    Expects flattened input x ∈ R^(C·H·W).
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int, t_dim: int):
        super().__init__()
        self.d_t = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size),
        )
        self.input_layer   = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.time_projs    = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.out_layer     = nn.Linear(hidden_size, input_size)
        self.act           = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e_t = self.time_mlp(sincos_embed(t, self.d_t))
        h   = self.act(self.input_layer(x))
        for layer, proj in zip(self.hidden_layers, self.time_projs):
            h = self.act(layer(h) + proj(e_t))
        return self.out_layer(h)
    
#  MLP_Residual  (feedforward with residual skip connections)
#
#  Same as MLP but with a skip connection at every hidden layer:
#    h = h + SiLU( Linear(h) + time_proj(e_t) )
#
#
#  The forward pass is:
#    (B, C*H*W)
#    input_layer: Linear(input_size, hidden_size)
#    for each hidden layer:
#        h = h + SiLU( Linear(h) + time_proj(e_t) )
#    out_layer: Linear(hidden_size, output_size)
#    (B, C*H*W)
    
class MLP_Residual(nn.Module):
    """
    we have the input size which is the flattened image
    the hidden_sizes are the sizes of the hidden layers
    the t_size it the size we want the time variable to occupy in the initial vector
    we will output a vector where if we put in R^n with time t in R^m, the model
    MLP: R^n times R^m to R^n so it will output back the image
    """
    def __init__(self, input_size, hidden_size, amount_layers, output_size, time_dimension: int):
        super().__init__()

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