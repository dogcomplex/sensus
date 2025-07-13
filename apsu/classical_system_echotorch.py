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

class ClassicalSystemEchoTorch:
    """
    Represents the physical substrate being controlled, using EchoTorch.

    This class encapsulates two Echo State Networks (ESNs), representing
    the "slow medium" of the experiment. It is designed to be a classical,
    deterministic (given a seed), high-dimensional dynamical system.
    This is a reimplementation of the `ClassicalSystem` class to use
    the EchoTorch library instead of reservoirpy.
    """
    def __init__(self, N=100, sr=0.95, lr=0.3, noise_rc=0.001, seed=1234, input_dim=1, device='cpu'):
        """
        Initializes the ClassicalSystem using EchoTorch.

        Args:
            N (int): Number of units in each reservoir (state vector dimension).
            sr (float): Spectral radius of the reservoir weight matrices.
            lr (float): Leaking rate of the reservoir neurons.
            noise_rc (float): Amount of internal noise in the reservoir.
            seed (int): Random seed for reproducibility.
            input_dim (int): The dimension of the input signal.
            device (str or torch.device): The device to run the simulation on.
        """
        self.N = N
        self.sr = sr
        self.lr = lr
        self.noise_rc = noise_rc
        self.seed = seed
        self.input_dim = input_dim
        self.device = torch.device(device)

        # Set seed for reproducibility
        echotorch.utils.manual_seed(seed)

        # 1. Create weight matrix generators
        w_generator = mg.NormalMatrixGenerator(
            spectral_radius=sr,
            connectivity=1.0 # Fully connected
        )
        win_generator = mg.NormalMatrixGenerator(mean=0.0, std=1.0)
        wbias_generator = mg.NormalMatrixGenerator(mean=0.0, std=1.0)
        
        # 2. Define a noise generator function that creates tensors on the correct device
        def noise_generator(size):
            return torch.randn(size, device=self.device) * noise_rc

        # 3. Create two identical reservoirs (A and B)
        # Reservoir A
        echotorch.utils.manual_seed(seed)
        w_A = w_generator.generate(size=(N, N), dtype=torch.float32).to(self.device)
        win_A = win_generator.generate(size=(N, input_dim), dtype=torch.float32).to(self.device)
        wbias_A = wbias_generator.generate(size=(N,), dtype=torch.float32).to(self.device)
        self.reservoir_A = LiESNCell(
            input_dim=input_dim,
            output_dim=N,
            w=w_A,
            w_in=win_A,
            w_bias=wbias_A,
            leaky_rate=lr,
            noise_generator=noise_generator,
            dtype=torch.float32
        ).to(self.device)

        # Reservoir B
        echotorch.utils.manual_seed(seed + 1) # Use a different seed for B
        w_B = w_generator.generate(size=(N, N), dtype=torch.float32).to(self.device)
        win_B = win_generator.generate(size=(N, input_dim), dtype=torch.float32).to(self.device)
        wbias_B = wbias_generator.generate(size=(N,), dtype=torch.float32).to(self.device)
        self.reservoir_B = LiESNCell(
            input_dim=input_dim,
            output_dim=N,
            w=w_B,
            w_in=win_B,
            w_bias=wbias_B,
            leaky_rate=lr,
            noise_generator=noise_generator,
            dtype=torch.float32
        ).to(self.device)

        # 4. Create readouts (but do not train them yet)
        self.readout_A = RRCell(input_dim=N, output_dim=1, dtype=torch.float32, device=self.device)
        self.readout_B = RRCell(input_dim=N, output_dim=1, dtype=torch.float32, device=self.device)

        self.states_A = []
        self.states_B = []

        # Ensure reservoirs are in training mode to update hidden states
        self.reservoir_A.train()
        self.reservoir_B.train()

    def reset(self):
        """Resets the readouts and collected states for a new trial."""
        self.readout_A.reset()
        self.readout_B.reset()
        self.states_A = []
        self.states_B = []

    def collect_state(self, state_A, state_B):
        """Appends states to internal storage for later readout training."""
        self.states_A.append(state_A)
        self.states_B.append(state_B)

    def train_readouts(self, targets_A, targets_B):
        """
        Trains the Ridge readouts on all collected states using the
        EchoTorch two-stage training process.
        """
        # Ensure readouts are in training mode
        self.readout_A.train(True)
        self.readout_B.train(True)
        
        # --- Move to CPU for memory-intensive training ---
        # The covariance matrix calculation can exhaust GPU VRAM.
        # Moving to CPU lets us use system RAM. We also up-cast to float64
        # for better numerical stability during this one-off training step.
        self.readout_A.to('cpu').to(torch.float64)
        self.readout_B.to('cpu').to(torch.float64)
        
        # Concatenate the collected states.
        X_A = torch.cat(self.states_A, dim=0).to('cpu', dtype=torch.float64)
        X_B = torch.cat(self.states_B, dim=0).to('cpu', dtype=torch.float64)

        # Convert numpy targets to tensors with shape (batch, time, features)
        y_A = torch.from_numpy(targets_A).double().unsqueeze(0).to('cpu')
        y_B = torch.from_numpy(targets_B).double().unsqueeze(0).to('cpu')

        # Stage 1: Accumulate xTx and xTy matrices
        self.readout_A(X_A, y_A)
        self.readout_B(X_B, y_B)

        # Stage 2: Finalize training to compute w_out
        self.readout_A.finalize()
        self.readout_B.finalize()
        
        # --- Move back to original device for inference ---
        self.readout_A.to(self.device).to(torch.float32)
        self.readout_B.to(self.device).to(torch.float32)

        # Diagnose the training by calculating the Mean Squared Error
        # The readout is now in eval mode after finalize()
        pred_A = self.readout_A(torch.cat(self.states_A, dim=0))
        pred_B = self.readout_B(torch.cat(self.states_B, dim=0))

        mse_A = torch.mean((pred_A.to(self.device) - torch.from_numpy(targets_A).float().unsqueeze(0).to(self.device)) ** 2).item()
        mse_B = torch.mean((pred_B.to(self.device) - torch.from_numpy(targets_B).float().unsqueeze(0).to(self.device)) ** 2).item()

        return mse_A, mse_B

    def get_readout_outputs(self):
        """
        Runs the collected states through the trained readouts to get
        the final output streams.
        """
        # Concatenate the collected states, which already have the correct shape.
        X_A = torch.cat(self.states_A, dim=0)
        X_B = torch.cat(self.states_B, dim=0)

        outputs_A = self.readout_A(X_A)
        outputs_B = self.readout_B(X_B)

        # Return as a simple numpy array
        return outputs_A.squeeze(0).detach().cpu().numpy(), outputs_B.squeeze(0).detach().cpu().numpy()

    def diagnose(self, steps=2000, save_path="apsu/diagnostics_report_echotorch.png"):
        """
        Runs a pre-flight check to validate reservoir health.
        """
        print(f"Running diagnosis for {steps} steps...")
        
        # 1. Run reservoirs with white-noise input.
        white_noise = (torch.randn(1, steps, self.input_dim) * 0.5).float().to(self.device)
        
        self.reservoir_A.reset_hidden()
        internal_states_A = self.reservoir_A(white_noise, reset_state=True)
        internal_states_A = internal_states_A.squeeze(0).detach().cpu().numpy()
        
        print("Diagnosis run complete. Generating plots...")

        # 3. Generate and save a multi-panel plot.
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"ClassicalSystem (EchoTorch) Diagnostic Report (Reservoir A)\nN={self.N}, sr={self.sr}, lr={self.lr}, noise={self.noise_rc}", fontsize=16)

        # Panel A: Histogram of activation values
        axes[0, 0].hist(internal_states_A.flatten(), bins=50, density=True)
        axes[0, 0].set_title("Neuron Activation Histogram")
        axes[0, 0].set_xlabel("Activation Value")
        axes[0, 0].set_ylabel("Density")

        # Panel B: Time-series of 5 random neurons
        random_neurons_indices = np.random.choice(self.N, 5, replace=False)
        for i in random_neurons_indices:
            axes[0, 1].plot(internal_states_A[:500, i], label=f"Neuron {i}")
        axes[0, 1].set_title("Time-series of 5 Random Neurons (First 500 steps)")
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Activation")
        axes[0, 1].legend(fontsize='small')

        # Panel C: 2D PCA projection of the attractor
        pca = PCA(n_components=2)
        states_pca = pca.fit_transform(internal_states_A)
        axes[1, 0].plot(states_pca[:, 0], states_pca[:, 1], lw=0.5)
        axes[1, 0].set_title("State Space Attractor (2D PCA Projection)")
        axes[1, 0].set_xlabel("Principal Component 1")
        axes[1, 0].set_ylabel("Principal Component 2")

        # Panel D: Blank for now, can add Reservoir B diagnostics later
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Reservoir B Diagnostics TBD', ha='center', va='center', fontsize=12)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        try:
            plt.savefig(save_path)
            print(f"Diagnostic report saved to {save_path}")
        except Exception as e:
            print(f"Error saving diagnostic plot: {e}")

        plt.close()


def main():
    """
    Main function to execute Phase 0: Baseline Characterization.
    """
    print("--- Starting Project Apsu (EchoTorch): Phase 0 ---")
    
    # Instantiate the system with default parameters from the spec, explicitly on CPU for diagnosis
    classical_system = ClassicalSystemEchoTorch(device='cpu')
    
    # Run the diagnostic pre-flight check
    classical_system.diagnose()
    
    print("--- Phase 0 Complete ---")


if __name__ == "__main__":
    main() 