import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'software', 'lib', 'EchoTorch')))

import torch
import echotorch
import echotorch.utils.matrix_generation as mg
from echotorch.nn.reservoir import LiESNCell
from echotorch.nn.linear import RRCell
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging

def create_leaky_esn_cell(units, spectral_radius, leaking_rate, noise_rc, seed, device):
    """
    Creates a single LeakyESN cell.
    """
    dtype = torch.float32
    if seed is not None:
        echotorch.utils.manual_seed(seed)

    w_generator = mg.NormalMatrixGenerator(spectral_radius=spectral_radius, connectivity=1.0)
    win_generator = mg.NormalMatrixGenerator(mean=0.0, std=1.0)
    wbias_generator = mg.NormalMatrixGenerator(mean=0.0, std=1.0)

    def noise_generator(size):
        return torch.randn(size, device=device) * noise_rc if noise_rc > 0 else None

    w = w_generator.generate(size=(units, units), dtype=dtype).to(device)
    w_in = win_generator.generate(size=(units, 1), dtype=dtype).to(device)
    w_bias = wbias_generator.generate(size=(units,), dtype=dtype).to(device)

    return LiESNCell(
        input_dim=1,
        output_dim=units,
        w=w,
        w_in=w_in,
        w_bias=w_bias,
        leaky_rate=leaking_rate,
        noise_generator=noise_generator if noise_rc > 0 else None,
        dtype=dtype
    ).to(device)


class ClassicalSystemEchoTorch:
    """
    Represents the physical substrate being controlled, using EchoTorch.
    """
    def __init__(self, units=100, spectral_radius=0.95, leaking_rate=0.3, noise_rc=0.001, input_scaling=1.0, seed=None, device='cuda'):
        """
        Initializes the ClassicalSystem using EchoTorch.
        """
        self.units = units
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.noise_rc = noise_rc
        self.input_scaling = input_scaling
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        # Create two identical reservoirs (A and B)
        self.reservoir_A = create_leaky_esn_cell(
            self.units, self.spectral_radius, self.leaking_rate, self.noise_rc, self.seed, self.device
        )
        self.reservoir_B = create_leaky_esn_cell(
            self.units, self.spectral_radius, self.leaking_rate, self.noise_rc, self.seed + 1 if self.seed is not None else None, self.device
        )

        # Create readouts
        self.readout_A = RRCell(input_dim=self.units, output_dim=1, device=self.device)
        self.readout_B = RRCell(input_dim=self.units, output_dim=1, device=self.device)

        # Initialize state tracking lists
        self.reset()

    def reset(self):
        """Resets the system to a clean state for a new trial."""
        if hasattr(self.reservoir_A, 'reset_hidden'):
            self.reservoir_A.reset_hidden()
            self.reservoir_B.reset_hidden()
        
        self.readout_A.reset()
        self.readout_B.reset()

        self.collected_states_A = []
        self.collected_states_B = []
        self.readout_outputs_A = None
        self.readout_outputs_B = None

    def step(self, input_A, input_B, collect=False):
        """
        Runs one step of the reservoirs and optionally collects the states.
        """
        input_A_tensor = input_A.reshape(1, 1, -1).to(self.device) * self.input_scaling
        input_B_tensor = input_B.reshape(1, 1, -1).to(self.device) * self.input_scaling
        
        self.reservoir_A(input_A_tensor)
        self.reservoir_B(input_B_tensor)

        if collect:
            self.collect_state(self.reservoir_A.hidden.clone(), self.reservoir_B.hidden.clone())
            
        return self.reservoir_A.hidden.clone(), self.reservoir_B.hidden.clone()

    def collect_state(self, x_A, x_B):
        self.collected_states_A.append(x_A.squeeze().cpu().detach())
        self.collected_states_B.append(x_B.squeeze().cpu().detach())

    def get_state(self):
        """Returns the concatenated current state of both reservoirs."""
        state_A = self.reservoir_A.hidden
        state_B = self.reservoir_B.hidden
        return torch.cat((state_A.flatten(), state_B.flatten()), dim=0).unsqueeze(0)

    def get_state_individual(self):
        """Returns the current state of each reservoir individually."""
        return self.reservoir_A.hidden.clone(), self.reservoir_B.hidden.clone()

    def train_readouts(self, targets_A, targets_B):
        """
        Trains the readout layers for both reservoirs on the collected states.
        """
        if not self.collected_states_A or not self.collected_states_B:
            raise ValueError("No states have been collected. Call step() with collect=True first.")

        states_A_tensor = torch.stack(self.collected_states_A, dim=0).unsqueeze(0).to(self.device)
        states_B_tensor = torch.stack(self.collected_states_B, dim=0).unsqueeze(0).to(self.device)

        # Robustly handle targets whether they are numpy arrays or tensors
        if isinstance(targets_A, np.ndarray):
            targets_A_tensor = torch.from_numpy(targets_A).float().view(-1, 1).to(self.device)
        else:
            targets_A_tensor = targets_A.float().view(-1, 1).to(self.device)

        if isinstance(targets_B, np.ndarray):
            targets_B_tensor = torch.from_numpy(targets_B).float().view(-1, 1).to(self.device)
        else:
            targets_B_tensor = targets_B.float().view(-1, 1).to(self.device)

        # Add a batch dimension to match the states tensor
        targets_A_tensor = targets_A_tensor.unsqueeze(0)
        targets_B_tensor = targets_B_tensor.unsqueeze(0)

        self.readout_A.reset()
        self.readout_A.train()
        self.readout_A(states_A_tensor, targets_A_tensor)
        self.readout_A.finalize()
        self.readout_A.eval()

        self.readout_B.reset()
        self.readout_B.train()
        self.readout_B(states_B_tensor, targets_B_tensor)
        self.readout_B.finalize()
        self.readout_B.eval()

        self.readout_outputs_A = self.readout_A(states_A_tensor)
        self.readout_outputs_B = self.readout_B(states_B_tensor)

        mse_A = torch.mean((self.readout_outputs_A - targets_A_tensor)**2).item()
        mse_B = torch.mean((self.readout_outputs_B - targets_B_tensor)**2).item()

        self.collected_states_A.clear()
        self.collected_states_B.clear()

        return mse_A, mse_B

    def get_readout_outputs(self):
        """Returns the outputs from the trained readout layers."""
        if self.readout_outputs_A is None or self.readout_outputs_B is None:
            raise ValueError("Readouts have not been trained or used yet.")
        return self.readout_outputs_A, self.readout_outputs_B

    def diagnose(self, steps=2000, plot_path="diagnostics_report.png", input_scaling=1.0):
        """
        Runs diagnostic checks on the reservoirs and generates a report.
        """
        logging.info(f"Starting diagnostic run for {steps} steps...")
        self.reset()
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(plot_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Generate random input signals for both reservoirs and apply scaling
        input_A = torch.randn(1, steps, 1, device=self.device) * input_scaling
        input_B = torch.randn(1, steps, 1, device=self.device) * input_scaling
        
        # Collect states
        states_A_diag = []
        states_B_diag = []

        with torch.no_grad():
            for i in range(steps):
                # We call the ESN cells directly to bypass the system's own input_scaling
                self.reservoir_A(input_A[:, i:i+1, :])
                self.reservoir_B(input_B[:, i:i+1, :])
                states_A_diag.append(self.reservoir_A.hidden.squeeze().cpu().numpy())
                states_B_diag.append(self.reservoir_B.hidden.squeeze().cpu().numpy())

        states_A_diag = np.array(states_A_diag)
        states_B_diag = np.array(states_B_diag)
        logging.info("Diagnostic run complete. Generating plots...")

        fig, axes = plt.subplots(2, 3, figsize=(22, 12), constrained_layout=True)
        fig.suptitle(f"Classical System Diagnostics Report (Units={self.units}, SR={self.spectral_radius}, LR={self.leaking_rate})", fontsize=16)
        
        self._plot_diagnosis(axes[0, :], states_A_diag, "Reservoir A", 'skyblue', 'darkviolet')
        self._plot_diagnosis(axes[1, :], states_B_diag, "Reservoir B", 'salmon', 'darkgreen')

        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Diagnostic report saved to {plot_path}")

    def _plot_diagnosis(self, ax_row, states, name, color1, color2):
        """Helper to plot diagnostic panel for one reservoir."""
        ax_row[0].hist(states.flatten(), bins=50, color=color1, edgecolor='black')
        ax_row[0].set_title(f"{name}: Activations Histogram")
        ax_row[0].set_xlabel("Activation Value")
        ax_row[0].set_ylabel("Frequency")

        for i in range(min(5, self.units)):
            neuron_idx = np.random.randint(0, states.shape[1])
            ax_row[1].plot(states[:200, neuron_idx], lw=1)
        ax_row[1].set_title(f"{name}: Random Neuron Activations (200 steps)")
        ax_row[1].set_xlabel("Time Step")
        ax_row[1].set_ylabel("Activation")

        if self.units > 1:
            pca = PCA(n_components=2)
            projected_states = pca.fit_transform(states)
            ax_row[2].plot(projected_states[:, 0], projected_states[:, 1], lw=0.5, color=color2)
            ax_row[2].set_title(f"{name}: PCA of State Space Attractor")
            ax_row[2].set_xlabel("Principal Component 1")
            ax_row[2].set_ylabel("Principal Component 2")
        else:
            ax_row[2].axis('off')
            ax_row[2].text(0.5, 0.5, 'PCA not applicable (units=1)', ha='center', va='center') 