import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from .classical_system_echotorch import ClassicalSystemEchoTorch
from .experiment_echotorch import run_chsh_trial_echotorch

class ZeroController(nn.Module):
    """
    A controller that always outputs zero, as required by the Phase 1 spec.
    This provides the baseline measurement for an uncontrolled system.
    """
    def forward(self, state_a, state_b):
        """
        The forward pass returns zero-tensors with the correct shape and device.
        
        Args:
            state_a (torch.Tensor): The state from reservoir A.
            state_b (torch.Tensor): The state from reservoir B.

        Returns:
            (torch.Tensor, torch.Tensor): A tuple of zero-valued corrective signals.
        """
        # The shape of the output must match the expected correction shape,
        # which is [sequence_length, 1].
        seq_len = state_a.shape[0]
        device = state_a.device
        c_a = torch.zeros((seq_len, 1), device=device, dtype=state_a.dtype)
        c_b = torch.zeros((seq_len, 1), device=device, dtype=state_b.dtype)
        return c_a, c_b

def run_phase1_null_experiment(num_trials=100, device_name='cuda'):
    """
    Executes the Phase 1 Null Experiment for the EchoTorch implementation.
    """
    print("--- Starting Project Apsu (EchoTorch): Phase 1 (Null Experiment) ---")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    s_scores = []
    
    # Instantiate the components once. The system is the same for all trials,
    # only the random CHSH settings sequence (determined by the seed) changes.
    system = ClassicalSystemEchoTorch(N=100, device=device)
    controller = ZeroController().to(device)

    for i in tqdm(range(num_trials), desc="Running CHSH Trials"):
        # The seed for the trial itself changes. This tests the CHSH measurement
        # apparatus against different random input sequences.
        s = run_chsh_trial_echotorch(controller, system, seed=i, device=device, delay=1)
        s_scores.append(s)

    s_scores = np.array(s_scores)
    mean_s = np.mean(s_scores)
    std_s = np.std(s_scores)

    print("\n--- Null Experiment Results ---")
    print(f"Number of trials: {num_trials}")
    print(f"Mean S-score: {mean_s:.4f}")
    print(f"Std. Dev of S-scores: {std_s:.4f}")
    print(f"Max S-score observed: {np.max(s_scores):.4f}")

    # Plotting the distribution of results
    plt.figure(figsize=(10, 6))
    plt.hist(s_scores, bins=20, density=True, alpha=0.8, label=f'Mean: {mean_s:.3f} ± {std_s:.3f}')
    plt.axvline(2.0, color='r', linestyle='--', linewidth=2, label="Classical Bound (S=2)")
    plt.title("Distribution of S-Scores for Null (Zero) Controller (EchoTorch)")
    plt.xlabel("S-Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "apsu/phase1_null_experiment_results_echotorch.png"
    plt.savefig(save_path)
    print(f"\nResults histogram saved to {save_path}")
    plt.close()

    # Check against Success Gate C1 from spec §1.3
    if mean_s <= 2.0 and np.max(s_scores) < 2.02:
        print("\nSuccess Gate PASSED: The baseline classical system correctly obeys the classical bound.")
    else:
        print("\nSuccess Gate FAILED: The baseline system appears to violate the classical bound. Check for bugs.")

    print("--- Phase 1 (EchoTorch) Complete ---")

if __name__ == "__main__":
    run_phase1_null_experiment() 