import torch
import torch.nn as nn

class NonLocalCoordinator_small(nn.Module):
    """
    A "minimal" version of the Non-Local Coordinator, as per the
    Phase 2 review board's recommendation (Step 1: Simplify the Problem).

    This controller has a much smaller hidden layer (16 units vs 256),
    dramatically reducing the parameter count to create a smoother,
    lower-dimensional search space for the optimizer.
    """
    def __init__(self, esn_dimension, hidden_dim=16, is_linear=False):
        """
        Args:
            esn_dimension (int): The state dimension (N) of a single ESN.
            hidden_dim (int): Dimension of the hidden layer. Fixed at 16.
            is_linear (bool): If True, creates a single-layer linear controller.
        """
        super(NonLocalCoordinator_small, self).__init__()
        
        input_dim = 2 * esn_dimension
        output_dim = 2 # Corrective signal c_A and c_B

        if is_linear:
            model_uncompiled = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Tanh()
            )
        else:
            model_uncompiled = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh()
            )
        
        self.model = model_uncompiled
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Applies Kaiming He uniform initialization to linear layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state_a, state_b):
        """
        Computes the corrective signals.
        """
        combined_state = torch.cat([state_a, state_b], dim=-1)
        corrections = self.model(combined_state)
        c_a = corrections[:, 0].unsqueeze(-1)
        c_b = corrections[:, 1].unsqueeze(-1)
        return c_a, c_b 