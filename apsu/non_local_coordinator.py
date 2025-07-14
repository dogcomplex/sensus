import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NonLocalCoordinator(nn.Module):
    """
    The "Fast Controller" - a non-linear function approximator (NN).
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2, use_bias=True):
        super(NonLocalCoordinator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=use_bias),
                nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=use_bias),
                nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=use_bias),
                nn.Tanh()
            )
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the network using Kaiming He uniform initialization.
        """
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        """
        Performs the forward pass of the controller.

        Args:
            x (torch.Tensor): The combined state of the two reservoirs.

        Returns:
            torch.Tensor: The corrective signals, bounded between [-1, 1].
        """
        return self.network(x)

    def get_n_params(self):
        """Returns the total number of trainable parameters in the controller."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_weights(self, weights: np.ndarray):
        """
        Sets the weights of the controller from a flat numpy array.
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"Weights must be a numpy array, but got {type(weights)}")
            
        with torch.no_grad():
            start = 0
            for param in self.parameters():
                if param.requires_grad:
                    n_param = param.numel()
                    end = start + n_param
                    # Ensure the weights tensor has the same dtype and device as the model parameter.
                    new_data = torch.from_numpy(weights[start:end]).reshape(param.shape)
                    param.data = new_data.to(device=param.device, dtype=param.dtype)
                    start = end 