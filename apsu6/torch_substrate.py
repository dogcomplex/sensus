import torch
import torch.nn as nn
from typing import Optional

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
    state: Optional[torch.Tensor]

    def __init__(self, input_dim, units, sr=0.9, lr=0.1, connectivity=0.1, input_scaling=1.0, seed=42, device='cpu'):
        super().__init__()
        self.units = units
        self.lr = lr
        self.device = device
        
        self.Win = _uniform(units, input_dim, seed=seed, device=self.device) * input_scaling
        
        W = _sparse_normal(units, units, connectivity, seed=seed+1, device=self.device)
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W).abs()
            W *= sr / torch.max(eigenvalues)
        self.W = W
        
        self.state = None
        # The 'to' method will be called by the parent module to move the state.

    def forward(self, input_step) -> torch.Tensor:
        """
        Performs a single state update (one time step).
        Args:
            input_step (torch.Tensor): The input for this step, shape (batch, features).
        """
        # Assign to a local variable so the JIT compiler can reason about its type
        state = self.state
        # Prove to the compiler that 'state' is not None in this path
        if state is None:
            # This branch should not be taken in practice due to the harness logic,
            # but it's necessary for the JIT compiler's static analysis.
            raise RuntimeError("Reservoir state has not been initialized. Call reset(batch_size) first.")

        # Ensure state has the same dtype as the weights
        state_dtype = self.W.dtype
        state = state.to(state_dtype)

        pre_activation = state @ self.W.T + input_step @ self.Win.T
        new_state = (1 - self.lr) * state + self.lr * torch.tanh(pre_activation)
        self.state = new_state
            
        return new_state

    def reset(self, batch_size: int = 1):
        """Resets the reservoir's hidden state to zeros."""
        self.state = torch.zeros(batch_size, self.units, device=self.device)

    def to(self, device):
        """Moves the entire module to the specified device."""
        self.device = device
        self.Win = self.Win.to(device)
        self.W = self.W.to(device)
        if self.state is not None:
            self.state = self.state.to(device)
        return super().to(device) 