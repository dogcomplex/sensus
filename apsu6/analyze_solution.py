import cma
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import os

from apsu6.harness import ExperimentHarness

def main():
    # Set the environment variable for deterministic CuBLAS operations.
    # This must be done *before* any CUDA operations are initialized.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Enforce deterministic calculations for a stable, final analysis.
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser(description="Analyze the best solution from a completed CMA-ES run.")
    parser.add_argument('results_dir', type=str, help="Path to the experiment results directory (e.g., 'apsu6/results/mannequin_...').")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    config_path = results_path / "config.json"
    
    # --- Load Config ---
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find config.json in {results_path}")
        return
    
    weights_path = results_path / "best_controller_weights.npy"
    if not weights_path.exists():
        print(f"Error: Could not find best_controller_weights.npy in {results_path}")
        return

    print(f"--- Loading Best Solution from {weights_path} ---")
    best_solution = np.load(weights_path)
    
    print("\n--- Re-evaluating Best Solution ---")
    
    # We need to run a single evaluation to get the diagnostics.
    # The harness will be created with the loaded config.
    config['evaluation']['num_avg'] = 256 # Use a high num_avg for an accurate final score
    harness = ExperimentHarness(config)
    
    # We are not interested in the fitness, just the diagnostics
    s_score, diagnostics = harness.evaluate_fitness(best_solution, sensor_noise_std=0.0)

    print("\n--- Analysis Complete ---")
    print(f"Best S-Score: {s_score:.6f}")
    
    print("\nFull Diagnostics:")
    # Pretty print the diagnostics dictionary, converting tuple keys to strings for JSON compatibility.
    diagnostics['correlations'] = {str(k): v for k, v in diagnostics['correlations'].items()}
    print(json.dumps(diagnostics, indent=2))

    # --- Extract and display the discovered substrate parameters ---
    if config.get("anneal_substrate", False):
        print("\n--- Discovered Substrate Hyperparameters ---")
        controller_dim = sum(p.numel() for p in harness.temp_controller.parameters())
        substrate_hyperparams = best_solution[controller_dim:]
        sr_a = np.clip(substrate_hyperparams[0], 0.1, 1.5)
        lr_a = np.clip(substrate_hyperparams[1], 0.1, 1.0)
        sr_b = np.clip(substrate_hyperparams[2], 0.1, 1.5)
        lr_b = np.clip(substrate_hyperparams[3], 0.1, 1.0)
        print(f"  - sr_A: {sr_a:.6f}")
        print(f"  - lr_A: {lr_a:.6f}")
        print(f"  - sr_B: {sr_b:.6f}")
        print(f"  - lr_B: {lr_b:.6f}")


if __name__ == "__main__":
    main() 