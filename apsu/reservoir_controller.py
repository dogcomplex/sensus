import torch
import torch.nn as nn
import numpy as np

# Import the builder for the ESN cell from the classical system
from .classical_system_echotorch import create_leaky_esn_cell

class ReservoirController(nn.Module):
    """
    A controller that uses an Echo State Network (ESN) as its core.

    Instead of a feed-forward neural network, this controller uses the rich,
    dynamic state of a fixed, random reservoir to compute the corrective signals.
    The optimization process trains a linear 'readout' layer on top of this
    reservoir's state.
    """
    def __init__(self, input_dim=200, output_dim=2, units=50, spectral_radius=1.1, 
                 leaking_rate=0.3, noise_rc=0.001, seed=None, **kwargs):
        """
        Initializes the ReservoirController.

        Args:
            input_dim (int): The dimension of the input from the main system.
                             Note: The internal ESN is hardcoded to 1D input steps.
            output_dim (int): The dimension of the corrective signal (usually 2).
            units (int): The number of neurons in the controller's ESN.
            spectral_radius (float): The spectral radius of the ESN's weight matrix.
            leaking_rate (float): The leaking rate of the ESN's neurons.
            noise_rc (float): Internal noise level for the ESN.
            seed (int): Random seed for reproducibility.
        """
        super(ReservoirController, self).__init__()
        
        # Keep track of the device
        self.device = None # Will be set by the first .to(device) call
        
        # Instantiate the ESN cell that will act as our controller's "brain"
        # We create it on CPU first and move it later.
        self.reservoir = create_leaky_esn_cell(
            units=units,
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            noise_rc=noise_rc,
            seed=seed,
            device='cpu' 
        )
        
        # The only trainable part: a linear readout layer.
        # It maps the reservoir's state to the output control signals.
        self.readout = nn.Linear(units, output_dim, bias=True)
        
        # Final output bounding
        self.output_activation = torch.tanh

        self.init_weights()
    
    def to(self, *args, **kwargs):
        """Override .to() to keep track of the device."""
        new_self = super().to(*args, **kwargs)
        # Infer device from the first argument if it's a torch.device or string
        if len(args) > 0:
            if isinstance(args[0], (torch.device, str)):
                new_self.device = torch.device(args[0])
            # Handle other cases if necessary
        return new_self

    def init_weights(self):
        """Initializes the readout weights."""
        # A simple uniform initialization is fine for the linear readout
        nn.init.uniform_(self.readout.weight, -0.1, 0.1)
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def forward(self, x):
        """
        Performs the forward pass of the controller.
        
        Args:
            x (torch.Tensor): The combined state of the two main reservoirs.
                               Shape: (batch, features), e.g., (1, 200)
            
        Returns:
            torch.Tensor: The corrective signals, bounded between [-1, 1].
        """
        if self.device is None:
            self.device = x.device
        
        # The ESNCell expects a time-series input. We have a single state vector.
        # We can treat the features of the state vector as a sequence of inputs
        # to drive the controller reservoir for a few steps.
        # This allows the reservoir to develop a more complex internal state.
        
        # x has shape (1, 200). Reshape to (1, 200, 1) to feed as a sequence.
        reservoir_input = x.unsqueeze(-1)
        
        # Run the reservoir. It updates its hidden state internally.
        self.reservoir(reservoir_input)
        
        # The new state is in reservoir.hidden. Use this for the readout.
        reservoir_state = self.reservoir.hidden
        
        # Pass the final reservoir state through the linear readout
        output = self.readout(reservoir_state)
        
        # Apply final bounding activation
        return self.output_activation(output)

    def reset(self):
        """Resets the reservoir's hidden state for a new trial."""
        if hasattr(self.reservoir, 'reset_hidden'):
            self.reservoir.reset_hidden()

    def get_n_params(self):
        """Returns the total number of trainable parameters in the controller."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_weights(self, weights: np.ndarray):
        """
        Sets the weights of the controller from a flat numpy array.
        In this controller, this only affects the final readout layer.
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
