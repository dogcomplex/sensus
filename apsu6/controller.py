import torch
import torch.nn as nn
from math import sqrt

class UniversalController(nn.Module):
    """
    A polymorphic controller that can be configured to emulate different
    physical realities (Quantum, Mannequin, Absurdist), as specified in the
    v6.0 requirements document.
    """
    def __init__(self, protocol: str, N_A: int, N_B: int, K_controller: int, 
                 R_speed: float, signaling_bits: int, 
                 internal_cell_config: dict, device: torch.device):
        """
        Initializes the controller for a specific protocol.

        Args:
            protocol: 'Quantum', 'Mannequin', or 'Absurdist'.
            N_A: The dimension of substrate A's state space.
            N_B: The dimension of substrate B's state space.
            K_controller: A parameter controlling complexity (e.g., hidden layer size).
            R_speed: The speed ratio, used for R>1 internal loops.
            signaling_bits: For Protocol A, the bandwidth of the signaling channel.
            internal_cell_config: Config for the R>1 recurrent cell.
            device: The torch compute device.
        """
        super().__init__()
        self.protocol = protocol
        self.device = device
        self.R = R_speed
        self.signaling_bits = signaling_bits
        self.internal_cell_config = internal_cell_config

        self._build_network(N_A, N_B, K_controller)
        
        if self.R > 1 and self.internal_cell_config.get('enabled', False):
            cell_type = self.internal_cell_config['type']
            hidden_size = self.internal_cell_config['hidden_size']
            
            if self.protocol == 'Mannequin':
                cell_input_size = N_A + N_B
            else:
                cell_input_size = N_A + N_B + 2
            
            if cell_type == 'gru':
                self.internal_cell = nn.GRUCell(cell_input_size, hidden_size)
            else:
                raise NotImplementedError(f"Internal cell type '{cell_type}' not supported.")
                
            self.internal_decoder = nn.Linear(hidden_size, K_controller)

        self.to(self.device)

    def _build_network(self, N_A: int, N_B: int, K_controller: int):
        """Helper method to construct the appropriate NN architecture."""
        if self.protocol == 'Quantum':
            self.quantum_state_dim = 4
        
        elif self.protocol == 'Mannequin':
            # This protocol uses a shared encoder for the substrate states (x_A, x_B)
            # and, if R>1, a separate encoder for the internal "thought" vector.
            # This avoids dynamic layer sizing and is a cleaner design.
            self.substrate_encoder = nn.Sequential(
                nn.Linear(N_A + N_B, K_controller),
                nn.ReLU()
            )
            # The head's input size will be consistent now.
            # It receives the encoded substrate and potentially an encoded thought.
            head_input_dim = K_controller
            
            if self.R > 1 and self.internal_cell_config.get('enabled', False):
                self.thought_encoder = nn.Sequential(
                    nn.Linear(K_controller, K_controller),
                    nn.ReLU()
                )
                head_input_dim += K_controller # Add space for the thought vector

            self.head_A = nn.Sequential(
                nn.Linear(head_input_dim + N_A + 1, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, 1),
                nn.Tanh() # Bound the output for stability
            )
            self.head_B = nn.Sequential(
                nn.Linear(head_input_dim + N_B + 1, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, 1),
                nn.Tanh() # Bound the output for stability
            )

        elif self.protocol == 'Absurdist':
            mlp_input_size = N_A + N_B + 2 + self.signaling_bits
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_size, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, K_controller),
                nn.ReLU(),
                nn.Linear(K_controller, 2),
                nn.Tanh()
            )
        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")
    
    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor, 
                settings_A: torch.Tensor, settings_B: torch.Tensor) -> torch.Tensor:
        """The main forward pass, dispatched based on protocol."""
        
        internal_iterations = max(1, int(round(self.R))) if self.R > 1 else 1
        if internal_iterations > 1 and hasattr(self, 'internal_cell'):
            if self.protocol == 'Mannequin':
                 cell_input = torch.cat([x_A, x_B], dim=-1)
            else:
                 cell_input = torch.cat([x_A, x_B, settings_A, settings_B], dim=-1)

            hidden_state = torch.zeros(x_A.size(0), self.internal_cell.hidden_size, device=self.device)
            
            # --- Generous Thinking Loop ---
            # Collect the hidden state at every step of the internal loop.
            all_hidden_states = []
            for _ in range(internal_iterations):
                hidden_state = self.internal_cell(cell_input, hidden_state)
                all_hidden_states.append(hidden_state)

            # Use max-pooling to find the "strongest" thought across all steps.
            # This allows a "moment of clarity" to drive the final decision.
            stacked_states = torch.stack(all_hidden_states, dim=0)
            pooled_hidden_state, _ = torch.max(stacked_states, dim=0)
            
            thought_vector = self.internal_decoder(pooled_hidden_state)
        else:
            thought_vector = None

        if self.protocol == 'Quantum':
            raise NotImplementedError("Protocol Q is not yet implemented.")
            
        elif self.protocol == 'Mannequin':
            # Always encode the raw substrate state
            substrate_latent = self.substrate_encoder(torch.cat([x_A, x_B], dim=-1))
            
            # If a "thought" was generated, encode it and concatenate
            if thought_vector is not None and hasattr(self, 'thought_encoder'):
                thought_latent = self.thought_encoder(thought_vector)
                shared_latent = torch.cat([substrate_latent, thought_latent], dim=-1)
            else:
                shared_latent = substrate_latent

            input_A = torch.cat([shared_latent, x_A, settings_A], dim=-1)
            correction_A = self.head_A(input_A)
            input_B = torch.cat([shared_latent, x_B, settings_B], dim=-1)
            correction_B = self.head_B(input_B)
            
        elif self.protocol == 'Absurdist':
            raise NotImplementedError("Protocol A is not yet fully implemented.")
            # signal_vec = self._encode_signal(x_A, settings_A)
            # controller_input = torch.cat([x_A, x_B, settings_A, settings_B, signal_vec], dim=-1)
            # corrections = self.mlp(controller_input) 
            # correction_A, correction_B = corrections[..., 0:1], corrections[..., 1:2]

        return torch.cat([correction_A, correction_B], dim=-1)

    def _encode_signal(self, x, s):
        """Placeholder for a learned, deterministic, bandwidth-limited signal encoder."""
        return torch.zeros(x.size(0), self.signaling_bits, device=self.device)

    def reset(self):
        """Resets any internal state of the controller."""
        if self.protocol == 'Quantum':
            raise NotImplementedError("Protocol Q reset is not yet implemented.")
        # For other protocols, state is managed in the forward pass, so no reset needed.
        pass 