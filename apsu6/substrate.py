import torch
from apsu6.torch_substrate import StatefulReservoir

class ClassicalSubstrate:
    """
    Encapsulates the two Echo State Network (ESN) reservoirs representing the
    "slow medium" of the experiment. This implementation uses a pure PyTorch
    backend to allow for full GPU acceleration and stateful, step-by-step evaluation.
    """
    def __init__(self, N_A: int, N_B: int, sr_A: float, sr_B: float, 
                 lr_A: float, lr_B: float, seed_A: int, seed_B: int, 
                 device: torch.device, **kwargs):
        self.device = device
        self.N_A, self.N_B = N_A, N_B
        self.input_dim = 2 # [setting, correction]
        
        reservoir_params = {'sr': sr_A, 'lr': lr_A, 'seed': seed_A, 'device': device}
        self.reservoir_A = StatefulReservoir(self.input_dim, N_A, **reservoir_params)
        
        reservoir_params = {'sr': sr_B, 'lr': lr_B, 'seed': seed_B, 'device': device}
        self.reservoir_B = StatefulReservoir(self.input_dim, N_B, **reservoir_params)
        
    def step(self, input_A: torch.Tensor, input_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evolves the batch of systems by one time step.
        Args:
            input_A (torch.Tensor): Shape (batch, features)
            input_B (torch.Tensor): Shape (batch, features)
        """
        # .forward() is now stateful and performs one step
        state_A = self.reservoir_A(input_A)
        state_B = self.reservoir_B(input_B)
        return state_A, state_B

    def get_current_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the current internal states of the reservoirs."""
        if self.reservoir_A.state is None or self.reservoir_B.state is None:
            raise RuntimeError("Reservoir states are not initialized. Call reset() first.")
        return self.reservoir_A.state, self.reservoir_B.state

    def reset(self, batch_size=1):
        """
        Resets the internal state of both reservoirs for a new batch.
        """
        self.reservoir_A.reset(batch_size)
        self.reservoir_B.reset(batch_size)

    def to(self, device):
        """Moves the substrate and its components to the specified device."""
        self.device = device
        self.reservoir_A.to(device)
        self.reservoir_B.to(device)
        return self

    def half(self):
        """Converts internal reservoirs to half precision."""
        self.reservoir_A.half()
        self.reservoir_B.half()
        return self

    def diagnose(self):
        """
        Performs pre-flight checks and generates diagnostic plots.
        (Note: Diagnostics may need to be adapted for torch tensors).
        """
        print("Diagnostics for Reservoir A:")
        print(self.reservoir_A)
        print("\nDiagnostics for Reservoir B:")
        print(self.reservoir_B)
        pass 