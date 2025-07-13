import torch
import torch.nn as nn

class NonLocalCoordinator(nn.Module):
    """
    The Non-Local Coordinator (NLC).

    As per spec §4.2, this is the trainable controller module. For the first
    part of Phase 2, this is a simple linear controller to test baseline
    capabilities.

    It takes the concatenated state of both ESNs and computes a corrective
    signal for each.
    """
    def __init__(self, esn_dimension, hidden_dim=None, is_linear=True):
        """
        Args:
            esn_dimension (int): The state dimension (N) of a single ESN.
                                 The input to the NLC is 2*N.
            hidden_dim (int, optional): Dimension of the hidden layer. If None,
                                        a linear controller is created.
            is_linear (bool): If True, creates a single-layer linear controller.
                              This overrides hidden_dim.
        """
        super(NonLocalCoordinator, self).__init__()
        
        input_dim = 2 * esn_dimension
        output_dim = 2 # Corrective signal c_A and c_B

        if is_linear:
            # Per Phase 2 spec, start with a simple linear controller.
            self.model = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                # Tanh is a critical safety mechanism to bound outputs to [-1, 1]
                # and prevent runaway feedback (Spec §4.2)
                nn.Tanh()
            )
        else:
            # Full non-linear controller for later phases (Spec §4.2)
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh()
            )
        
        # Initialize weights as per spec §4.2
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

        Can handle both single state vectors [1, N] and sequences [T, N].

        Args:
            state_a (torch.Tensor): The state vector(s) of ESN A.
            state_b (torch.Tensor): The state vector(s) of ESN B.

        Returns:
            (torch.Tensor, torch.Tensor): Corrective signal(s) c_a, c_b.
        """
        # Combine along the feature dimension
        # Input shape for a sequence: state_a=[T, 100], state_b=[T, 100]
        # Output shape of combined_state: [T, 200]
        combined_state = torch.cat([state_a, state_b], dim=-1)
        
        # Get corrective signals. The model processes the sequence directly.
        # Output shape of corrections: [T, 2]
        corrections = self.model(combined_state)
        
        # Split the corrections back into c_a and c_b
        # Output shapes: [T, 1] for both
        c_a = corrections[:, 0].unsqueeze(-1)
        c_b = corrections[:, 1].unsqueeze(-1)

        return c_a, c_b 