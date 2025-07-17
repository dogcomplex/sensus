import torch
import torch.nn as nn

def _uniform(rows, cols, seed, device):
    """Initializes a uniform random tensor on the specified device."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(rows, cols, generator=generator, device=device) * 2 - 1

def _normal(rows, cols, seed, device):
    """Initializes a normal random tensor on the specified device."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(rows, cols, generator=generator, device=device)

def _sparse_normal(rows, cols, connectivity, seed, device):
    """Initializes a sparse normal random tensor on the specified device."""
    t = _normal(rows, cols, seed, device)
    mask = torch.rand(rows, cols, device=device) > connectivity
    t[mask] = 0
    return t

class StatefulReservoir(nn.Module):
    """
    A stateful, step-by-step PyTorch implementation of an Echo State Network reservoir.
    This module manages its own hidden state from one step to the next.
    """
    def __init__(self, input_dim, units, sr=0.9, lr=0.1, connectivity=0.1, input_scaling=1.0, seed=42, device='cpu'):
        super().__init__()
        self.units = units
        self.lr = lr
        self.device = device
        
        win_tensor = _uniform(units, input_dim, seed=seed, device=self.device) * input_scaling
        self.Win = nn.Parameter(win_tensor, requires_grad=False)
        
        w_tensor = _sparse_normal(units, units, connectivity, seed=seed+1, device=self.device)
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(w_tensor).abs()
            w_tensor *= sr / torch.max(eigenvalues)
        self.W = nn.Parameter(w_tensor, requires_grad=False)
        
        self.state = None
        # The 'to' method will be called by the parent module to move the state.

    def forward(self, input_step):
        """
        Performs a single state update (one time step).
        Args:
            input_step (torch.Tensor): The input for this step, shape (batch, features).
        """
        if self.state is None:
            raise RuntimeError("Reservoir state has not been initialized. Call reset(batch_size) first.")

        # Ensure state has the same dtype as the weights
        state_dtype = self.W.dtype
        state = self.state.to(state_dtype)

        pre_activation = state @ self.W.T + input_step @ self.Win.T
        new_state = (1 - self.lr) * state + self.lr * torch.tanh(pre_activation)
        self.state = new_state
            
        return self.state

    def reset(self, batch_size=1):
        """Resets the reservoir's hidden state to zeros."""
        # Infer dtype from the weight matrix to support half-precision.
        dtype = self.W.dtype
        self.state = torch.zeros(batch_size, self.units, device=self.device, dtype=dtype)

    def to(self, device):
        """Moves the entire module to the specified device."""
        self.device = device
        self.Win = self.Win.to(device)
        self.W = self.W.to(device)
        if self.state is not None:
            self.state = self.state.to(device)
        return super().to(device)

    def half(self):
        """Converts weights and state to half precision for performance."""
        # Manually convert non-parameter/buffer tensors
        if self.state is not None:
            self.state = self.state.half()
        # The superclass's half() method will correctly handle
        # all registered nn.Parameter attributes (like self.Win and self.W).
        return super().half() 