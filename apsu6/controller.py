import torch
import torch.nn as nn
from apsu6.minimal_reservoir import MinimalESN as ESN

class UniversalController(nn.Module):
    """
    A polymorphic controller that can be configured to emulate different
    physical realities (Quantum, Mannequin, Absurdist).
    """
    def __init__(self, protocol: str, N_A: int, N_B: int, K_controller: int, 
                 R_speed: float, signaling_bits: int, 
                 internal_cell_config: dict, architecture: str = "mlp", device: torch.device = 'cpu'):
        super().__init__()
        self.protocol = protocol
        self.device = device
        self.R = R_speed
        self.signaling_bits = signaling_bits
        self.internal_cell_config = internal_cell_config
        self.architecture = architecture

        self._build_network(N_A, N_B, K_controller)
        self.to(self.device)

    def _build_network(self, N_A: int, N_B: int, K_controller: int):
        """Helper method to construct the appropriate NN architecture."""
        if self.protocol != 'Mannequin':
            raise NotImplementedError(f"Protocol '{self.protocol}' is not yet supported.")

        # --- Shared Encoder ---
        if self.architecture == "cross_product":
            encoder_input_size = N_A * N_B
            self.shared_encoder = nn.Sequential(
                nn.Linear(encoder_input_size, K_controller),
                nn.ReLU()
            )
        elif self.architecture == "esn":
            # For the ESN controller, the "encoder" is the ESN itself.
            esn_units = self.internal_cell_config.get('hidden_size', K_controller)
            self.esn_controller = ESN(
                input_dim=N_A + N_B,
                hidden_dim=esn_units,
                output_dim=K_controller, # The ESN's output is our shared latent vector
                spectral_radius=self.internal_cell_config.get('sr', 1.1),
                leaky_rate=self.internal_cell_config.get('lr', 0.5),
                input_scaling=self.internal_cell_config.get('input_scaling', 0.9),
                device=self.device
            )
            # The ESN class in EchoTorch includes the readout, so shared_encoder is not needed.
            self.shared_encoder = None
        else: # Default MLP architecture
            encoder_input_size = N_A + N_B
            self.shared_encoder = nn.Sequential(
                nn.Linear(encoder_input_size, K_controller),
                nn.ReLU()
            )

        # --- Internal Recurrent Cell (for R > 1) ---
        self.internal_gru = None
        self.internal_decoder = None
        head_input_dim = K_controller
        if self.R > 1 and self.internal_cell_config.get('enabled', False):
            hidden_size = self.internal_cell_config.get('hidden_size', K_controller)
            num_layers = self.internal_cell_config.get('num_layers', 1)
            
            self.internal_gru = nn.GRU(
                input_size=K_controller, 
                hidden_size=hidden_size, 
                num_layers=num_layers,
                batch_first=False
            )
            self.internal_decoder = nn.Linear(hidden_size, K_controller)
            head_input_dim = K_controller * 2 # substrate_latent + thought_latent

        # --- Output Heads ---
        self.head_A = nn.Sequential(
            nn.Linear(head_input_dim + N_A + 1, K_controller), # combined_latent + x_a + setting_a
            nn.ReLU(),
            nn.Linear(K_controller, 1),
            nn.Tanh()
        )
        self.head_B = nn.Sequential(
            nn.Linear(head_input_dim + N_B + 1, K_controller), # combined_latent + x_b + setting_b
            nn.ReLU(),
            nn.Linear(K_controller, 1),
            nn.Tanh()
        )

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor, 
                settings_A: torch.Tensor, settings_B: torch.Tensor,
                h_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        # --- 1. Substrate Encoding ---
        if self.architecture == "cross_product":
            # x_A is (B, N_A), x_B is (B, N_B) -> outer_product is (B, N_A, N_B)
            outer_product = torch.bmm(x_A.unsqueeze(2), x_B.unsqueeze(1))
            encoder_input = outer_product.view(outer_product.size(0), -1)
            substrate_latent = torch.relu(self.shared_encoder(encoder_input))
        elif self.architecture == "esn":
            # EchoTorch runs natively on GPU tensors, no CPU round-trip needed.
            encoder_input = torch.cat([x_A, x_B], dim=-1)
            # The ESN handles its own state internally.
            substrate_latent, _ = self.esn_controller(encoder_input)
        else: # Default MLP
            encoder_input = torch.cat([x_A, x_B], dim=-1)
            substrate_latent = torch.relu(self.shared_encoder(encoder_input))

        # --- 2. Internal "Thinking" Loop (R > 1) ---
        h_out = h_prev
        if self.R > 1 and self.internal_gru is not None:
            cell_input = substrate_latent.unsqueeze(0)
            gru_output, h_out = self.internal_gru(cell_input, h_prev)
            thought_latent = torch.relu(self.internal_decoder(gru_output.squeeze(0)))
            combined_latent = torch.cat([substrate_latent, thought_latent], dim=-1)
        else:
            combined_latent = substrate_latent

        # --- 3. Output Calculation ---
        input_A = torch.cat([combined_latent, x_A, settings_A], dim=-1)
        correction_A = self.head_A(input_A)

        input_B = torch.cat([combined_latent, x_B, settings_B], dim=-1)
        correction_B = self.head_B(input_B)
        
        return torch.cat([correction_A, correction_B], dim=-1), h_out

    def reset(self):
        """Resets the internal hidden state of the GRU or ESN controller."""
        if hasattr(self, 'esn_controller'):
            self.esn_controller.reset_hidden()
        self.h_state = None 