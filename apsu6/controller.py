import torch
import torch.nn as nn
from typing import Tuple, Optional

class UniversalController(nn.Module):
    def __init__(self, protocol: str, N_A: int, N_B: int, K_controller: int, 
                 R_speed: float, signaling_bits: int, 
                 internal_cell_config: dict, device: torch.device):
        super().__init__()
        self.protocol = protocol
        self.device = device
        self.R = R_speed

        # Unroll the dict for JIT compatibility
        self.use_thinking_loop = self.R > 1 and internal_cell_config.get('enabled', False)

        self._build_network(N_A, N_B, K_controller, internal_cell_config)
        self.to(self.device)

    def _build_network(self, N_A: int, N_B: int, K_controller: int, internal_cell_config: dict):
        if self.protocol != 'Mannequin':
            raise NotImplementedError("Only Protocol M is supported for this controller version.")

        self.substrate_encoder = nn.Linear(N_A + N_B, K_controller)

        if self.use_thinking_loop:
            hidden_size = internal_cell_config.get('hidden_size', K_controller)
            self.internal_cell = nn.GRUCell(K_controller, hidden_size)
            self.internal_decoder = nn.Linear(hidden_size, K_controller)
            head_input_dim = K_controller * 2 # substrate_latent + thought_latent
        else:
            # Attributes must be defined in both branches for JIT
            self.internal_cell = None
            self.internal_decoder = None
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
                thought_h_in: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The main forward pass for a single time step.
        """
        substrate_state = torch.cat([x_A, x_B], dim=-1)
        substrate_latent = torch.relu(self.substrate_encoder(substrate_state))
        
        thought_h_out: Optional[torch.Tensor] = None
        if self.use_thinking_loop:
            # Assertions to help the JIT compiler
            assert self.internal_cell is not None
            assert self.internal_decoder is not None

            cell_input = substrate_latent
            hidden_state = thought_h_in if thought_h_in is not None else \
                         torch.zeros(x_A.size(0), self.internal_cell.hidden_size, device=self.device)
            
            thought_h_out = self.internal_cell(cell_input, hidden_state)
            thought_latent = torch.relu(self.internal_decoder(thought_h_out))
            shared_latent = torch.cat([substrate_latent, thought_latent], dim=-1)
        else:
            shared_latent = substrate_latent

        input_A = torch.cat([shared_latent, x_A, settings_A], dim=-1)
        correction_A = self.head_A(input_A)

        input_B = torch.cat([shared_latent, x_B, settings_B], dim=-1)
        correction_B = self.head_B(input_B)
            
        correction_logits = torch.cat([correction_A, correction_B], dim=-1)
        return correction_logits, thought_h_out

    def reset(self):
        pass 