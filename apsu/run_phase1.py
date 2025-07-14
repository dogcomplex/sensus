import sys
import os
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure the module is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apsu.chsh import evaluate_fitness

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_phase1(config_path):
    """
    Executes Phase 1: The Null Experiment.
    
    This phase rigorously tests the measurement apparatus by running the CHSH
    fitness evaluation with a "zero" controller that provides no corrective
    feedback. The resulting distribution of S-scores should be centered at or
    below the classical limit of 2.0.

    Args:
        config_path (str): Path to the JSON configuration file for Phase 1.
    """
    logging.info("--- Starting Project Apsu: Phase 1 (Null Experiment) ---")

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in configuration file: {config_path}")
        return

    phase1_config = config.get('phase1_config', {})
    num_trials = phase1_config.get('num_trials', 100)
    plot_path = phase1_config.get('plot_path', 'apsu/phase1_null_experiment_results.png')
    device = phase1_config.get('device', 'cpu')

    # --- 2. Execution ---
    chsh_seed_base = config.get('chsh_evaluation', {}).get('chsh_seed_base', 42)
    s_scores = []
    
    # Use tqdm for a progress bar
    for i in tqdm(range(num_trials), desc="Running Null Experiment Trials"):
        seed = chsh_seed_base + i
        # For the null experiment, the controller_weights are None.
        result = evaluate_fitness(
            weights=None, 
            config=config, 
            chsh_seed=seed,
            return_diagnostics=True
        )
        if result['s_value'] > -1:  # Check for valid result
            s_scores.append(result['s_value'])

    # --- 3. Analysis & Reporting ---
    s_scores = np.array(s_scores)
    
    if s_scores.size == 0:
        logging.error("All trials failed. No S-scores were generated.")
        mean_s = np.nan
        std_s = np.nan
    else:
    mean_s = np.mean(s_scores)
    std_s = np.std(s_scores)
    
    logging.info(f"Completed {len(s_scores)} trials.")
    logging.info(f"S-Score Mean: {mean_s:.4f}")
    logging.info(f"S-Score Std Dev: {std_s:.4f}")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(s_scores, bins=20, density=True, alpha=0.7, label=f'N={len(s_scores)} trials')
    ax.axvline(2.0, color='r', linestyle='--', label='Classical Limit (S=2.0)')
    ax.axvline(mean_s, color='k', linestyle=':', label=f'Mean S = {mean_s:.3f}')
    
    ax.set_xlabel("S-Score")
    ax.set_ylabel("Probability Density")
    ax.set_title("Phase 1: Null Experiment Results (Zero Controller)")
    ax.legend()
    
    output_dir = os.path.dirname(plot_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(plot_path)
    plt.close()
    
    logging.info(f"Results plot saved to {plot_path}")
    logging.info("--- Phase 1 Complete ---")
    
    # Success Gate Check
    if s_scores.size == 0:
        logging.error("VALIDATION FAILED: No successful trials.")
        sys.exit(1) # Exit with error code
    elif mean_s > 2.02:
        logging.warning(f"VALIDATION WARNING: Mean S-score ({mean_s:.4f}) is unexpectedly high for a null experiment.")
    else:
        logging.info("VALIDATION PASSED: Mean S-score is within the expected classical range.")
        
def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 null experiment for the Apsu project.")
    parser.add_argument(
        '--config',
        type=str,
        default='apsu/experiments/phase1/phase1_fast_config.json',
        help='Path to the JSON configuration file for Phase 1.'
    )
    args = parser.parse_args()
    
    run_phase1(args.config)

if __name__ == "__main__":
    main() 