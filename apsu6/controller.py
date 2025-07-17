import torch
import torch.nn as nn

class UniversalController(nn.Module):
    def __init__(self, protocol: str, N_A: int, N_B: int, K_controller: int, 
                 R_speed: float, signaling_bits: int, 
                 internal_cell_config: dict, device: torch.device):
        super().__init__()
        self.protocol = protocol
        self.device = device
        self.R = R_speed
        self.internal_cell_config = internal_cell_config

        self._build_network(N_A, N_B, K_controller)
        self.to(self.device)

    def _build_network(self, N_A: int, N_B: int, K_controller: int):
        if self.protocol != 'Mannequin':
            raise NotImplementedError("Only Protocol M is supported for this controller version.")

        self.substrate_encoder = nn.Linear(N_A + N_B, K_controller)

        if self.R > 1 and self.internal_cell_config.get('enabled', False):
            hidden_size = self.internal_cell_config.get('hidden_size', K_controller)
            num_layers = self.internal_cell_config.get('num_layers', 1)
            
            # Use a single, optimized GRU layer instead of a GRUCell loop in the harness
            self.internal_gru = nn.GRU(
                input_size=K_controller, 
                hidden_size=hidden_size, 
                num_layers=num_layers,
                batch_first=False # We process one time-step at a time, but GRU expects (seq, batch, feature)
            )
            self.internal_decoder = nn.Linear(hidden_size, K_controller)
            head_input_dim = K_controller * 2 # substrate_latent + thought_latent
        else:
            head_input_dim = K_controller # substrate_latent only

        self.head_A = nn.Sequential(
            nn.Linear(head_input_dim + N_A + 1, K_controller),
            nn.ReLU(),
            nn.Linear(K_controller, 1),
            nn.Tanh()
        )
        self.head_B = nn.Sequential(
            nn.Linear(head_input_dim + N_B + 1, K_controller),
            nn.ReLU(),
            nn.Linear(K_controller, 1),
            nn.Tanh()
        )
    
    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor, 
                settings_A: torch.Tensor, settings_B: torch.Tensor,
                thought_h_in: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The main forward pass for a single time step.
        Args:
            x_A, x_B: Current substrate states, shape (B, N_A/N_B)
            settings_A, settings_B: Current settings, shape (B, 1)
            thought_h_in: Optional incoming hidden state for the thinking loop.
        Returns:
            A tuple of (correction_logits, thought_h_out)
        """
        
        # Combine states for substrate encoder: (B, N_A+N_B)
        substrate_state = torch.cat([x_A, x_B], dim=-1)
        # Encode the current state: (B, N_A+N_B) -> (B, K)
        substrate_latent = torch.relu(self.substrate_encoder(substrate_state))
        
        thought_h_out = None # Default return for hidden state
        if self.R > 1 and self.internal_cell_config.get('enabled', False):
            # The "thinking" is now a single, highly optimized multi-layer GRU operation.
            # We add a dummy sequence dimension of 1.
            cell_input = substrate_latent.unsqueeze(0) # (1, B, K)
            
            # The harness is no longer responsible for the loop, only the hidden state.
            hidden_state = thought_h_in if thought_h_in is not None else \
                         torch.zeros(self.internal_gru.num_layers, x_A.size(0), self.internal_gru.hidden_size, device=self.device, dtype=cell_input.dtype)
            
            # The GRU layer handles the "thinking loop" internally in optimized code.
            gru_output, thought_h_out = self.internal_gru(cell_input, hidden_state)
            
            # We take the output of the final layer
            thought_latent = torch.relu(self.internal_decoder(gru_output.squeeze(0)))
            shared_latent = torch.cat([substrate_latent, thought_latent], dim=-1)
        else:
            shared_latent = substrate_latent

        # Combine all inputs for the final heads
        # Shapes: (B, K_shared), (B, N_A), (B, 1) -> (B, K_shared+N_A+1)
        input_A = torch.cat([shared_latent, x_A, settings_A], dim=-1)
        correction_A = self.head_A(input_A)

        input_B = torch.cat([shared_latent, x_B, settings_B], dim=-1)
        correction_B = self.head_B(input_B)
            
        correction_logits = torch.cat([correction_A, correction_B], dim=-1)
        return correction_logits, thought_h_out

    def reset(self):
        pass 