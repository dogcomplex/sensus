import re
import os
from pathlib import Path
import numpy as np

def recover_from_run_log(run_log_path):
    """Parses a single run.log file and extracts fitness data."""
    with open(run_log_path, 'r') as f:
        content = f.read()

    # Regex to extract the Best Fitness from each generation's log line
    fitness_regex = re.compile(r"Generation \d+: Best Fitness=([\d.-]+)")
    fitness_matches = fitness_regex.findall(content)

    if not fitness_matches:
        return None

    # Reconstruct the content for the cma_fit.dat file
    cma_log_data = ["% Cma-es logging file (recovered)"]
    cumulative_best = -np.inf
    for i, fitness_str in enumerate(fitness_matches):
        fitness = float(fitness_str)
        # The "best" fitness is the maximum found so far in that generation's population
        # and subsequent generations
        if fitness > cumulative_best:
            cumulative_best = fitness

        # CMA minimizes, so we negate the score for the log
        negated_fitness = -cumulative_best
        # Gen Evals Fitness Stdev AxisRatio ... BestFoundFitness
        # We only care about the 6th column (index 5)
        cma_log_data.append(f"  {i+1}  {(i+1)*10}  0.0  0.0  0.0  {negated_fitness}")
    
    return "\n".join(cma_log_data)

def main():
    base_results_dir = Path("apsu/experiments/goldilocks_sweep/results")
    if not base_results_dir.exists():
        print(f"ERROR: The base results directory was not found at '{base_results_dir}'")
        return

    print(f"--- Starting Recovery from run.log files in '{base_results_dir}' ---")
    recovered_files = 0
    
    for experiment_dir in base_results_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        run_log_file = experiment_dir / "run.log"
        config_file = experiment_dir / "config.json"
        cma_fit_file = experiment_dir / "cma_fit.dat"

        if run_log_file.exists() and config_file.exists() and not cma_fit_file.exists():
            print(f"Found missing log in: {experiment_dir}. Attempting recovery...")
            
            cma_data = recover_from_run_log(run_log_file)
            
            if cma_data:
                with open(cma_fit_file, 'w') as f:
                    f.write(cma_data)
                print(f"  -> Successfully recovered and wrote: {cma_fit_file}")
                recovered_files += 1
            else:
                print(f"  -> FAILED: No fitness data found in {run_log_file}")
        
    print(f"--- Recovery Complete. ---")
    print(f"Total files recovered: {recovered_files}")
    if recovered_files > 0:
        print("\nYou can now try running the plot_results.py script again.")
    else:
        print("\nNo missing log files were found or recovered. Data should be ready for plotting.")

if __name__ == "__main__":
    main() 