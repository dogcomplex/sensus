import torch
import torch.nn as nn

class PyTorchESN(nn.Module):
    """
    A pure-PyTorch implementation of a Leaky-Integrated Echo State Network (ESN)
    designed to act as a classical substrate. Its weights are initialized randomly
    but are exposed as parameters so they can be optimized by an external process
    (i.e., substrate annealing).
    """
    def __init__(self, input_dim: int, hidden_dim: int, sr: float, lr: float, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.leaky_rate = lr
        self.device = device

        # Initialize weights as trainable parameters. The optimizer will handle them.
        self.w_in = nn.Parameter(torch.randn(hidden_dim, input_dim))
        
        # Initialize reservoir weights and scale them once.
        w_res_raw = torch.randn(hidden_dim, hidden_dim)
        current_radius = torch.max(torch.abs(torch.linalg.eigvals(w_res_raw))).item()
        if current_radius > 1e-8:
            w_res_scaled = w_res_raw * (sr / current_radius)
        else:
            w_res_scaled = w_res_raw
        self.w_res = nn.Parameter(w_res_scaled)
        
        self.hidden_state = None
        self.to(device)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Performs a single step of the ESN update for a batch."""
        batch_size = u.size(0)
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_dim, device=self.device, dtype=u.dtype)

        pre_activation = u @ self.w_in.T + self.hidden_state @ self.w_res.T
        h_next = torch.tanh(pre_activation)
        self.hidden_state = (1 - self.leaky_rate) * self.hidden_state + self.leaky_rate * h_next
        return self.hidden_state

class ClassicalSubstrate(nn.Module):
    """
    A pure PyTorch implementation of the classical substrate, containing two ESNs.
    """
    def __init__(self, N_A: int, N_B: int, sr_A: float, sr_B: float, lr_A: float, lr_B: float, device: torch.device, **kwargs):
        super().__init__()
        self.N_A = N_A
        self.N_B = N_B
        
        # The input to each reservoir is [setting_bit, correction_signal]
        input_dim = 2
        
        self.reservoir_A = PyTorchESN(input_dim, N_A, sr=sr_A, lr=lr_A, device=device)
        self.reservoir_B = PyTorchESN(input_dim, N_B, sr=sr_B, lr=lr_B, device=device)
        self.to(device)

    def step(self, input_A: torch.Tensor, input_B: torch.Tensor):
        """Evolves both reservoirs by one time step."""
        self.reservoir_A(input_A)
        self.reservoir_B(input_B)
        # State is updated internally

    def get_current_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the current internal states of the reservoirs."""
        return self.reservoir_A.hidden_state, self.reservoir_B.hidden_state

    def reset(self, batch_size: int = 1):
        """Resets the internal state of both reservoirs."""
        dtype = next(self.parameters()).dtype
        self.reservoir_A.hidden_state = torch.zeros(batch_size, self.N_A, device=self.reservoir_A.device, dtype=dtype)
        self.reservoir_B.hidden_state = torch.zeros(batch_size, self.N_B, device=self.reservoir_B.device, dtype=dtype) 