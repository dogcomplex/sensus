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
    
    # --- Load Solutions ---
    solution_p1_path = results_path / "best_solution_phase1.npy"
    solution_final_path = results_path / "best_solution_final.npy"

    if not solution_p1_path.exists() or not solution_final_path.exists():
        print(f"Error: Could not find 'best_solution_phase1.npy' and/or 'best_solution_final.npy' in {results_path}")
        return

    print(f"\n--- Loading Phase 1 Solution from {solution_p1_path} ---")
    solution_p1 = np.load(solution_p1_path, allow_pickle=True)
    
    print(f"--- Loading Final Solution from {solution_final_path} ---")
    loaded_obj_final = np.load(solution_final_path, allow_pickle=True)

    # Handle the bug where the entire CMA object was saved instead of the solution vector.
    if loaded_obj_final.ndim == 0 and 'cma' in str(type(loaded_obj_final.item())):
        print("INFO: Detected corrupted final solution file. Extracting xbest from saved CMA object.")
        cma_es_object = loaded_obj_final.item()
        solution_final = cma_es_object.result.xbest
    else:
        solution_final = loaded_obj_final

    # --- Analysis ---
    
    # 1. Extract and display the discovered substrate parameters from Phase 1
    # We need a temporary harness to get the controller dimension
    temp_harness_for_dims = ExperimentHarness(config)
    controller_dim = sum(p.numel() for p in temp_harness_for_dims.temp_controller.parameters())
    del temp_harness_for_dims

    print("\n--- Discovered Substrate Hyperparameters (from Phase 1) ---")
    substrate_hyperparams = solution_p1[controller_dim:]
    sr_a = np.clip(substrate_hyperparams[0], 0.7, 1.5)
    lr_a = np.clip(substrate_hyperparams[1], 0.2, 1.0)
    sr_b = np.clip(substrate_hyperparams[2], 0.7, 1.5)
    lr_b = np.clip(substrate_hyperparams[3], 0.2, 1.0)
    print(f"  - sr_A: {sr_a:.6f}")
    print(f"  - lr_A: {lr_a:.6f}")
    print(f"  - sr_B: {sr_b:.6f}")
    print(f"  - lr_B: {lr_b:.6f}")

    # 2. Re-evaluate the FINAL solution with high precision
    print("\n--- Re-evaluating Final Solution (end_to_end) ---")
    
    # Prepare the config for the final evaluation
    # Lock in the discovered substrate parameters
    config['substrate_params']['sr_A'] = sr_a
    config['substrate_params']['lr_A'] = lr_a
    config['substrate_params']['sr_B'] = sr_b
    config['substrate_params']['lr_B'] = lr_b
    config['anneal_substrate'] = False # VERY IMPORTANT: we are no longer annealing
    
    # Use a high num_avg for an accurate final score and disable CPU fallback for precision
    config['evaluation']['num_avg'] = 256
    config['evaluation']['use_cpu_fallback_for_metrics'] = False
    
    harness = ExperimentHarness(config)
    
    # The final solution vector only contains controller weights
    s_score, diagnostics = harness.evaluate_fitness(solution_final, readout_mode='end_to_end')

    print("\n--- Analysis Complete ---")
    print(f"Definitive S-Score: {s_score:.6f}")
    
    print("\nFull Final Diagnostics:")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main() 