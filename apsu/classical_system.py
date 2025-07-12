import reservoirpy as rpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# Set a seed for reproducibility
rpy.set_seed(42)

class ClassicalSystem:
    """
    Represents the physical substrate being controlled, as per Apsu v4 spec.

    This class encapsulates two Echo State Networks (ESNs), representing
    the "slow medium" of the experiment. It is designed to be a classical,
    deterministic (given a seed), high-dimensional dynamical system.
    """
    def __init__(self, N=100, sr=0.95, lr=0.3, noise_rc=0.001, seed=1234):
        """
        Initializes the ClassicalSystem.

        Args:
            N (int): Number of units in each reservoir (state vector dimension).
            sr (float): Spectral radius of the reservoir weight matrices.
            lr (float): Leaking rate of the reservoir neurons.
            noise_rc (float): Amount of internal noise in the reservoir.
            seed (int): Random seed for reproducibility.
        """
        self.N = N
        self.sr = sr
        self.lr = lr
        self.noise_rc = noise_rc
        self.seed = seed

        # As per spec §4.1, create two identical reservoirs.
        self.reservoir_A = rpy.nodes.Reservoir(
            units=N,
            lr=lr,
            sr=sr,
            noise_rc=self.noise_rc,
            seed=seed
        )
        self.reservoir_B = rpy.nodes.Reservoir(
            units=N,
            lr=lr,
            sr=sr,
            noise_rc=self.noise_rc,
            seed=seed + 1 # Use a different seed for B for variety
        )
        
        # Per spec, create readouts but do not train them yet.
        self.readout_A = rpy.nodes.Ridge(output_dim=1)
        self.readout_B = rpy.nodes.Ridge(output_dim=1)

        self.states_A = []
        self.states_B = []

    def step(self, input_A, input_B):
        """Evolves the system by one time step."""
        state_A = self.reservoir_A.run(input_A)
        state_B = self.reservoir_B.run(input_B)
        return state_A, state_B

    def collect_state(self, state_A, state_B):
        """Appends states to internal storage for later readout training."""
        self.states_A.append(state_A)
        self.states_B.append(state_B)

    def train_readouts(self, targets_A, targets_B, alpha=1.0):
        """
        Trains the Ridge readouts on all collected states.

        This method performs a Ridge regression to find the optimal linear
        mapping from the collected high-dimensional states to the target
        binary outputs. This is crucial for a fair evaluation.

        Args:
            targets_A (np.ndarray): The target outputs for reservoir A.
            targets_B (np.ndarray): The target outputs for reservoir B.
            alpha (float): Regularization strength for the Ridge regression.

        Returns:
            (float, float): The mean squared error for readout A and B.
        """
        # We need to concatenate the collected states into a single matrix
        X_A = np.concatenate(self.states_A, axis=0)
        X_B = np.concatenate(self.states_B, axis=0)

        # Train the readout models
        self.readout_A.fit(X_A, targets_A)
        self.readout_B.fit(X_B, targets_B)

        # Diagnose the training by calculating the Mean Squared Error
        pred_A = self.readout_A.run(X_A)
        pred_B = self.readout_B.run(X_B)

        mse_A = np.mean((pred_A - targets_A) ** 2)
        mse_B = np.mean((pred_B - targets_B) ** 2)

        return mse_A, mse_B

    def get_readout_outputs(self):
        """
        Runs the collected states through the trained readouts to get
        the final output streams.
        """
        X_A = np.concatenate(self.states_A, axis=0)
        X_B = np.concatenate(self.states_B, axis=0)

        outputs_A = self.readout_A.run(X_A)
        outputs_B = self.readout_B.run(X_B)

        return outputs_A, outputs_B

    def diagnose(self, steps=2000, save_path="apsu/diagnostics_report.png"):
        """
        Runs a pre-flight check to validate reservoir health.

        As per spec §4.1, this generates a visual fingerprint of the
        reservoir's dynamics by running it with white-noise input and
        plotting key characteristics.

        Args:
            steps (int): The number of time steps to run the diagnosis.
            save_path (str): Path to save the output plot.
        """
        print(f"Running diagnosis for {steps} steps...")
        
        # 1. Run reservoirs with white-noise input.
        white_noise = np.random.randn(steps, 1) * 0.5
        
        internal_states_A = self.reservoir_A.run(white_noise, reset=True)
        
        print("Diagnosis run complete. Generating plots...")

        # 3. Generate and save a multi-panel plot.
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"ClassicalSystem Diagnostic Report (Reservoir A)\nN={self.N}, sr={self.sr}, lr={self.lr}, noise={self.noise_rc}", fontsize=16)

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
    print("--- Starting Project Apsu: Phase 0 ---")
    
    # Instantiate the system with default parameters from the spec
    classical_system = ClassicalSystem()
    
    # Run the diagnostic pre-flight check
    classical_system.diagnose()
    
    print("--- Phase 0 Complete ---")


if __name__ == "__main__":
    main() 