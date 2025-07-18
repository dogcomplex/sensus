import torch
import torch.nn as nn

class MinimalESN(nn.Module):
    """
    A minimal, pure-PyTorch implementation of a Leaky-Integrated Echo State Network (ESN).
    This version is designed to have its weights be fully trainable by an external optimizer.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 leaky_rate: float = 0.5, device: torch.device = 'cpu', **kwargs):
        """
        Initializes the ESN.

        Args:
            input_dim (int): The dimension of the input signal.
            hidden_dim (int): The number of neurons in the reservoir.
            output_dim (int): The dimension of the output signal from the readout.
            leaky_rate (float): The leaking rate (alpha) for the neuron state updates.
            device (torch.device): The computation device.
            **kwargs: Catches unused arguments like spectral_radius and input_scaling for API compatibility.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.leaky_rate = leaky_rate
        self.device = device

        # --- Initialize weights as TRAINABLE parameters ---
        # The optimizer will find the correct scaling and spectral radius.
        self.w_in = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.w_res = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.w_out = nn.Parameter(torch.randn(output_dim, hidden_dim))

        self.hidden_state = None
        self.to(self.device)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single step of the ESN update.

        Args:
            u (torch.Tensor): The input signal for the current time step, with shape (batch_size, input_dim).

        Returns:
            A tuple containing:
            - output (torch.Tensor): The output from the linear readout, shape (batch_size, output_dim).
            - self.hidden_state (torch.Tensor): The updated reservoir state, shape (batch_size, hidden_dim).
        """
        if u.dim() != 2:
            raise ValueError(f"Input tensor must be 2D (batch_size, input_dim), but got shape {u.shape}")

        batch_size = u.size(0)
        # Initialize hidden state if it's the first step or if batch size changes
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        # ESN update equation
        pre_activation = u @ self.w_in.T + self.hidden_state @ self.w_res.T
        h_next = torch.tanh(pre_activation)

        # Apply leaky integration
        self.hidden_state = (1 - self.leaky_rate) * self.hidden_state + self.leaky_rate * h_next

        # Apply the linear readout
        output = self.hidden_state @ self.w_out.T

        return output, self.hidden_state

    def reset_hidden(self):
        """Resets the internal hidden state of the reservoir."""
        self.hidden_state = None 