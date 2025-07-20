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
    # This must be done *before* any CUDA operations are initialized for it to work.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser(description="Analyze the best solution from a Grand Unified Strategy run.")
    parser.add_argument('results_dir', type=str, help="Path to the experiment results directory (e.g., 'apsu6/results/unified_strategy_...').")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    config_path = results_path / "config.json"
    
    # --- Load Config ---
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("--- Loaded Configuration ---")
        print(json.dumps(config, indent=2))
    except FileNotFoundError:
        print(f"Error: Could not find config.json in {results_path}")
        return
    
    # --- Load Solution ---
    solution_path = results_path / "best_solution_final.npy"
    if not solution_path.exists():
        print(f"Error: Could not find 'best_solution_final.npy' in {results_path}")
        return

    print(f"\n--- Loading Final Solution from {solution_path} ---")
    best_solution = np.load(solution_path, allow_pickle=True)

    # --- Analysis ---
    
    # 1. Extract and display the discovered substrate parameters
    if config.get("anneal_substrate", False):
        temp_harness_for_dims = ExperimentHarness(config)
        controller_dim = sum(p.numel() for p in temp_harness_for_dims.temp_controller.parameters())
        del temp_harness_for_dims

        print("\n--- Discovered Substrate Hyperparameters ---")
        substrate_hyperparams = best_solution[controller_dim:]
        # Use the config-driven clip ranges for accurate reporting
        sr_clip = config['substrate_params'].get('sr_clip_range', [0.7, 1.5])
        lr_clip = config['substrate_params'].get('lr_clip_range', [0.1, 1.0])
        sr_a = np.clip(substrate_hyperparams[0], sr_clip[0], sr_clip[1])
        lr_a = np.clip(substrate_hyperparams[1], lr_clip[0], lr_clip[1])
        sr_b = np.clip(substrate_hyperparams[2], sr_clip[0], sr_clip[1])
        lr_b = np.clip(substrate_hyperparams[3], lr_clip[0], lr_clip[1])
        print(f"  - sr_A: {sr_a:.6f}")
        print(f"  - lr_A: {lr_a:.6f}")
        print(f"  - sr_B: {sr_b:.6f}")
        print(f"  - lr_B: {lr_b:.6f}")

        # Lock in the discovered parameters for the re-evaluation run
        config['substrate_params']['sr_A'] = sr_a
        config['substrate_params']['lr_A'] = lr_a
        config['substrate_params']['sr_B'] = sr_b
        config['substrate_params']['lr_B'] = lr_b
    
    # 2. Re-evaluate the FINAL solution with high precision
    print("\n--- Re-evaluating Final Solution (end_to_end) ---")
    
    config['anneal_substrate'] = False # VERY IMPORTANT: we are no longer annealing
    config['evaluation']['num_avg'] = 256
    config['evaluation']['use_cpu_fallback_for_metrics'] = False
    
    harness = ExperimentHarness(config)
    
    s_score, diagnostics = harness.evaluate_fitness(best_solution)

    print("\n--- Analysis Complete ---")
    print(f"Definitive S-Score: {s_score:.6f}")
    
    print("\nFull Final Diagnostics:")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main() 