import reservoirpy as rpy
import numpy as np
import torch

class ClassicalSubstrate:
    """
    Encapsulates the two Echo State Network (ESN) reservoirs representing the
    "slow medium" of the experiment (Systems A and B). This implementation
    uses the 'reservoirpy' library, which is primarily NumPy-based.
    """
    def __init__(self, N_A: int, N_B: int, sr_A: float, sr_B: float, 
                 lr_A: float, lr_B: float, noise_A: float, noise_B: float, 
                 seed_A: int, seed_B: int, device: torch.device):
        """
        Initializes two distinct ESN reservoirs.
        """
        self.device = device
        self.N_A, self.N_B = N_A, N_B
        self.input_dim = 2
        
        self.reservoir_A = rpy.nodes.Reservoir(
            units=N_A, input_dim=self.input_dim, sr=sr_A, lr=lr_A, 
            noise_rc=noise_A, seed=seed_A
        )
        self.reservoir_B = rpy.nodes.Reservoir(
            units=N_B, input_dim=self.input_dim, sr=sr_B, lr=lr_B, 
            noise_rc=noise_B, seed=seed_B
        )
        
        self.reset()

    def step(self, input_A: np.ndarray, input_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evolves the system by one time step.
        """
        state_A = self.reservoir_A.run(input_A)
        state_B = self.reservoir_B.run(input_B)
        return state_A, state_B

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the current internal states of the reservoirs as detached torch
        tensors on the correct device, with a batch dimension added.
        """
        state_A_np = self.reservoir_A.state()
        state_B_np = self.reservoir_B.state()

        # Handle case where state is None on first call before any step
        if state_A_np is None:
            state_A_np = np.zeros((1, self.N_A))
        if state_B_np is None:
            state_B_np = np.zeros((1, self.N_B))
        
        state_A = torch.from_numpy(state_A_np).float().to(self.device)
        state_B = torch.from_numpy(state_B_np).float().to(self.device)
        
        if state_A.ndim == 1:
            state_A = state_A.unsqueeze(0)
        if state_B.ndim == 1:
            state_B = state_B.unsqueeze(0)
            
        return state_A, state_B

    def reset(self):
        """
        Resets the internal state of both reservoirs to a valid zero-state.
        """
        self.reservoir_A.reset()
        self.reservoir_B.reset()

    def diagnose(self):
        """
        Performs pre-flight checks and generates diagnostic plots.
        """
        print("Diagnostics for Reservoir A:")
        print(self.reservoir_A)
        print("\nDiagnostics for Reservoir B:")
        print(self.reservoir_B)
        pass 