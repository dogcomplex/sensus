import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from apsu.classical_system import ClassicalSystem
from apsu import chsh

# Parameters for the experiment as per spec §4.3
T_WASHOUT = 1000  # Steps to let the reservoir settle
T_EVAL = 4000     # Steps to evaluate, split into 4 blocks for each CHSH setting
N_TRIALS_PER_RUN = T_EVAL
N_RUNS = 100      # Number of full experiments to run for statistical robustness

def run_single_null_experiment(seed):
    """
    Executes a single, complete CHSH null experiment.

    In the null experiment, there is no Non-Local Coordinator. The CHSH
    setting choices are fed directly as input to the classical system.

    Args:
        seed (int): Random seed for this specific run.

    Returns:
        float: The final calculated S-score for this run.
    """
    # 1. Setup Phase
    system = ClassicalSystem(seed=seed)
    
    # Generate the stream of settings for Alice and Bob for this run
    alice_settings, bob_settings = chsh.get_chsh_settings(T_EVAL, seed=seed)
    
    # The inputs to the reservoirs are simply the settings {-1, 1}
    # We reshape to match the expected input shape for the ESNs
    inputs_A = (2 * alice_settings - 1).reshape(-1, 1)
    inputs_B = (2 * bob_settings - 1).reshape(-1, 1)
    
    # 2. Simulation Phase
    # First, a washout period with zero input to let the reservoir settle
    washout_input = np.zeros((T_WASHOUT, 1))
    system.step(washout_input, washout_input)

    # Then, the evaluation period where states are collected
    for i in range(T_EVAL):
        # The input at each step is the CHSH setting for that trial
        input_A_t = np.array([[inputs_A[i, 0]]])
        input_B_t = np.array([[inputs_B[i, 0]]])
        
        state_A, state_B = system.step(input_A_t, input_B_t)
        system.collect_state(state_A, state_B)

    # 3. Readout Training Phase
    # Generate the 'ideal' QM targets for the given settings
    targets_A, targets_B = chsh.get_chsh_targets(alice_settings, bob_settings)
    
    # Train the readouts to best fit the collected states to the targets
    system.train_readouts(targets_A, targets_B)
    
    # 4. Scoring Phase
    # Get the actual outputs from the trained readouts
    outputs_A, outputs_B = system.get_readout_outputs()

    # Calculate the S-score based on the system's actual behavior
    s_score = chsh.calculate_s_score(outputs_A, outputs_B, alice_settings, bob_settings)
    
    return s_score

def main():
    """
    Main function to execute Phase 1: The Null Experiment.
    """
    print("--- Starting Project Apsu: Phase 1 (Null Experiment) ---")
    
    s_scores = []
    # Use tqdm for a progress bar
    for i in tqdm(range(N_RUNS), desc="Running Null Experiments"):
        # Use a different seed for each run for statistical independence
        s_score = run_single_null_experiment(seed=i)
        s_scores.append(s_score)
        
    print(f"\nCompleted {N_RUNS} runs.")
    
    s_scores = np.array(s_scores)
    mean_s = np.mean(s_scores)
    std_s = np.std(s_scores)
    
    print(f"Mean S-score: {mean_s:.4f}")
    print(f"Std Dev S-score: {std_s:.4f}")
    
    # 5. Deliverable: Plot histogram of S-scores
    plt.figure(figsize=(10, 6))
    plt.hist(s_scores, bins=20, density=True, alpha=0.7, label="S-score Distribution")
    plt.axvline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.title(f"Distribution of S-Scores for Null Experiment (N={N_RUNS} runs)")
    plt.xlabel("S-Score")
    plt.ylabel("Density")
    plt.legend()
    
    save_path = "apsu/phase1_null_experiment_results.png"
    plt.savefig(save_path)
    print(f"Results histogram saved to {save_path}")
    plt.close()
    
    # Check Success Gate
    if mean_s <= 2.02: # Allow for a small margin of error
        print("\nSuccess Gate Passed: Mean S-score is within the classical bound.")
    else:
        print("\nSuccess Gate FAILED: Mean S-score significantly exceeds the classical bound.")
        
    print("--- Phase 1 Complete ---")

if __name__ == "__main__":
    main() 