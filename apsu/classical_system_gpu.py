"""
A PyTorch-based, GPU-accelerated implementation of the ClassicalSystem.

This module re-implements the necessary components from reservoirpy (which is
numpy-based) using PyTorch to allow the entire simulation to run on a GPU,
avoiding costly CPU-GPU data transfers at each simulation step.
"""
import torch
import torch.nn as nn

def _uniform(rows, cols, seed):
    """Initializes a uniform random tensor."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(rows, cols, generator=generator) * 2 - 1

def _normal(rows, cols, seed):
    """Initializes a normal random tensor."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator)

def _sparse_normal(rows, cols, connectivity, seed):
    """Initializes a sparse normal random tensor."""
    t = _normal(rows, cols, seed)
    mask = torch.rand(rows, cols) > connectivity
    t[mask] = 0
    return t

class Reservoir(nn.Module):
    """PyTorch implementation of an Echo State Network reservoir."""
    def __init__(self, input_dim, units, sr=0.9, lr=0.1, connectivity=0.1, input_scaling=1.0, seed=42):
        super().__init__()
        self.units = units
        self.lr = lr
        
        # Win: Input weights
        self.Win = _uniform(units, input_dim, seed=seed) * input_scaling
        
        # W: Recurrent weights
        W = _sparse_normal(units, units, connectivity, seed=seed+1)
        # Rescale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W).abs()
        W *= sr / torch.max(eigenvalues)
        self.W = W
        
    def run(self, input_sequence, initial_state=None):
        """Processes a sequence of inputs in a fully vectorized manner."""
        n_steps = input_sequence.shape[0]
        
        if initial_state is None:
            state = torch.zeros(1, self.units, device=input_sequence.device)
        else:
            state = initial_state
            
        history = torch.empty(n_steps, self.units, device=input_sequence.device)

        # This loop is now internal to the GPU-accelerated part of the code
        for i in range(n_steps):
            pre_activation = state @ self.W.T + input_sequence[i].unsqueeze(0) @ self.Win.T
            new_state = (1 - self.lr) * state + self.lr * torch.tanh(pre_activation)
            state = new_state
            history[i] = new_state

        return history, state # Return final state as well

    def to(self, device):
        self.Win = self.Win.to(device)
        self.W = self.W.to(device)
        return super().to(device)

class Ridge(nn.Module):
    """PyTorch implementation of Ridge regression for readout."""
    def __init__(self, input_dim, output_dim, ridge_param=1e-3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.ridge_param = ridge_param

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, Y):
        # Solves (X.T @ X + lambda*I) @ w = X.T @ Y
        # A = X.T @ X + lambda*I
        # b = X.T @ Y
        eye = torch.eye(X.shape[1], device=X.device)
        A = X.T @ X + self.ridge_param * eye
        b = X.T @ Y
        
        # weights = torch.linalg.solve(A, b) -> Not always stable
        # Use pseudo-inverse for stability
        weights = torch.linalg.pinv(A) @ b
        
        with torch.no_grad():
            self.linear.weight.data = weights.T
            # Bias is not trained in this formulation
            if self.linear.bias is not None:
                self.linear.bias.data.fill_(0)

class ClassicalSystemGPU:
    """Manages the ESNs and Readouts, all on the GPU."""
    def __init__(self, esn_dimension=100, seed=42, device='cpu'):
        self.device = device
        self.reservoir_A = Reservoir(1, esn_dimension, seed=seed).to(device)
        self.reservoir_B = Reservoir(1, esn_dimension, seed=seed+1000).to(device) # Different seed for B
        
        self.readout_A = Ridge(esn_dimension, 1).to(device)
        self.readout_B = Ridge(esn_dimension, 1).to(device)
        
        # State history is now generated on-the-fly by the run method
        self.states_A_history = None
        self.states_B_history = None

    def run_and_collect(self, inputs_A, inputs_B, washout_steps):
        # inputs_A and inputs_B are the full sequences for the evaluation run
        
        # 1. Washout phase (un-vectorized is fine, it's not the bottleneck)
        washout_input = torch.zeros((1, 1), device=self.device)
        
        # Initialize states explicitly
        state_A = torch.zeros(1, self.reservoir_A.units, device=self.device)
        state_B = torch.zeros(1, self.reservoir_B.units, device=self.device)

        for _ in range(washout_steps):
            pre_A = state_A @ self.reservoir_A.W.T + washout_input @ self.reservoir_A.Win.T
            state_A = (1 - self.reservoir_A.lr) * state_A + self.reservoir_A.lr * torch.tanh(pre_A)
            
            pre_B = state_B @ self.reservoir_B.W.T + washout_input @ self.reservoir_B.Win.T
            state_B = (1 - self.reservoir_B.lr) * state_B + self.reservoir_B.lr * torch.tanh(pre_B)

        # 2. Evaluation phase (vectorized)
        # Pass the final washout state as the initial state for the run
        self.states_A_history, _ = self.reservoir_A.run(inputs_A, initial_state=state_A)
        self.states_B_history, _ = self.reservoir_B.run(inputs_B, initial_state=state_B)

    def train_readouts(self, targets_A, targets_B):
        # targets are numpy arrays, convert them to tensors
        targets_A = torch.tensor(targets_A, dtype=torch.float32, device=self.device)
        targets_B = torch.tensor(targets_B, dtype=torch.float32, device=self.device)
        
        # The history tensors are already correctly shaped for training
        self.readout_A.fit(self.states_A_history, targets_A)
        self.readout_B.fit(self.states_B_history, targets_B)

    def get_readout_outputs(self):
        # Use the history tensors directly
        outputs_A = self.readout_A(self.states_A_history)
        outputs_B = self.readout_B(self.states_B_history)
        
        # Return as numpy arrays to interface with existing CHSH calculation
        return outputs_A.detach().cpu().numpy(), outputs_B.detach().cpu().numpy() 