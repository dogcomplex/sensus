import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_sweep_results():
    """
    Scans for experiment results, aggregates them, and plots S-Score vs. Controller Size.
    """
    results_dir = Path("apsu/standalone_results")
    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
        return

    logging.info(f"Scanning for results in: {results_dir}")

    results = []
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        results_file = exp_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    # Ensure the essential keys are in the dictionary
                    if 'controller_units' in data and 'best_s_score' in data:
                        results.append({
                            "units": data['controller_units'],
                            "score": data['best_s_score']
                        })
                    else:
                        logging.warning(f"Skipping malformed results file: {results_file}")
            except json.JSONDecodeError:
                logging.error(f"Could not decode JSON from {results_file}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while reading {results_file}: {e}")

    if not results:
        logging.warning("No valid results found to plot.")
        return

    # Sort results by controller size for cleaner plotting
    results.sort(key=lambda x: x['units'])
    
    units = [r['units'] for r in results]
    scores = [r['score'] for r in results]

    logging.info(f"Found {len(results)} data points: {results}")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(units, scores, marker='o', linestyle='-', color='b', label='Best S-Score')
    
    # Add a horizontal line for the classical limit
    ax.axhline(y=2.0, color='r', linestyle='--', label='Classical Limit (S=2)')
    # Add a horizontal line for the Tsirelson bound
    ax.axhline(y=2 * np.sqrt(2), color='g', linestyle='--', label="Tsirelson's Bound (S≈2.828)")


    ax.set_title('Controller Size vs. Best S-Score', fontsize=16, fontweight='bold')
    ax.set_xlabel('Controller Hidden Units', fontsize=12)
    ax.set_ylabel('Best Achieved S-Score', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Set x-axis ticks to be the actual unit sizes we tested
    ax.set_xticks(units)
    plt.xticks(rotation=45)
    
    # Set y-axis limits to be slightly more than the max score
    ax.set_ylim(0, max(scores) * 1.1 if scores else 4.0)

    plt.tight_layout()
    
    # Save the figure
    output_path = Path("sweep_results.png")
    plt.savefig(output_path, dpi=300)
    logging.info(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_sweep_results() 