import cma
import json
import argparse
from pathlib import Path
import numpy as np

from apsu6.harness import ExperimentHarness

def main():
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
    
    # CMA-ES saves its state in a directory named 'cma_es_out' by default
    # within the working directory at the time of the run. Since our main
    # script runs from the root, it should be in the root.
    # We need to find the specific file corresponding to the run.
    # The default logger name is 'cma_es_logger'.
    # This part is tricky as we don't know the exact run name.
    # Let's assume for now the user can find the .pkl file.
    # A more robust solution would be to save the CMA output path in the main script.
    
    print("--- Loading CMA-ES Data ---")
    print("Please locate the '.pkl' file from the CMA-ES output.")
    # For now, we will assume a default path structure for the analysis
    cma_output_dir = Path("cma_es_out")
    if not cma_output_dir.exists():
        print(f"Error: Default CMA output directory '{cma_output_dir}' not found.")
        print("CMA-ES might have saved its data elsewhere. You may need to specify the path.")
        return

    # Find the most recent pickle file in the output directory
    try:
        es_files = sorted(cma_output_dir.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
        latest_es_file = es_files[0]
        print(f"Found latest CMA-ES state file: {latest_es_file}")
    except IndexError:
        print(f"Error: No .pkl files found in {cma_output_dir}")
        return

    es = cma.CMAEvolutionStrategy.load(latest_es_file)
    best_solution = es.result.xbest

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
    # Pretty print the diagnostics dictionary
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