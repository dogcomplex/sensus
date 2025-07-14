import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
import re
import json
import numpy as np

def parse_cma_log(log_path):
    """Parses a CMA-ES log file to extract generation and best fitness."""
    data = []
    # cma logs whitespace-delimited data, starting with a comment char '%'
    # Columns are: Gen, Evals, Fitness, Stdev, AxisRatio, ... BestFoundFitness
    # We want the 6th column (index 5) for BestFoundFitness.
    with open(log_path, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) > 5:
                try:
                    gen = int(parts[0])
                    # CMA minimizes, so we store the negative to get the actual S-Score
                    fitness = float(parts[5]) * -1.0
                    data.append({'generation': gen, 'best_fitness': fitness})
                except (ValueError, IndexError):
                    continue
    return pd.DataFrame(data)

def get_config_delay(config_path):
    """Reads a config file and extracts the delay."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Navigate through the potentially nested structure
            if 'chsh_evaluation' in config:
                return config['chsh_evaluation'].get('delay', 'N/A')
            return config.get('delay', 'N/A')
    except (FileNotFoundError, json.JSONDecodeError):
        return 'N/A'


def plot_s_curve(results_df, output_path):
    """Plots the final S-score vs. Controller Delay (d)."""
    # Get the max fitness for each delay
    final_scores = results_df.loc[results_df.groupby('delay')['best_fitness'].idxmax()]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.lineplot(data=final_scores, x='delay', y='best_fitness', marker='o', ax=ax, markersize=10, lw=2.5)
    ax.axhline(2.0, color='red', linestyle='--', label='Classical Bound (S=2)')
    ax.axhline(2 * np.sqrt(2), color='green', linestyle='--', label="Tsirelson's Bound (S≈2.828)")
    
    ax.set_title('S(R) Curve: Best CHSH Score vs. Controller Delay', fontsize=18)
    ax.set_xlabel('Controller Delay, d (ticks)', fontsize=14)
    ax.set_ylabel('Best Found S-Score', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    
    # Annotate points
    for _, row in final_scores.iterrows():
        ax.text(row['delay'], row['best_fitness'], f" {row['best_fitness']:.4f}", ha='center', va='bottom')
        
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"S(R) curve plot saved to {output_path}")

def plot_combined_progress(results_df, output_path):
    """Plots all optimization progresses on a single graph."""
    # Calculate the cumulative max fitness for each group
    results_df['cumulative_best'] = results_df.groupby('delay')['best_fitness'].cummax()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    sns.lineplot(data=results_df, x='generation', y='cumulative_best', hue='delay', palette='viridis', lw=2, ax=ax)
    
    ax.set_title('Cumulative Best Fitness vs. Generation', fontsize=18)
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Best S-Score Found', fontsize=14)
    ax.legend(title='Delay (d)')
    ax.grid(True, which='both', linestyle='--')
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined progress plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot summary results from Apsu experiments.")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='apsu/experiments/cma_es/results_full',
        help='Directory containing the experiment result folders.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='apsu/review/summary',
        help='Directory to save the summary plots.'
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    
    # Match directories that look like they belong to the s_curve run
    for experiment_dir in results_path.iterdir():
        if experiment_dir.is_dir() and 'apsu_experiment' in experiment_dir.name:
            log_file = experiment_dir / 'cma_fit.dat'
            config_file = experiment_dir / 'config.json'
            
            if log_file.exists() and config_file.exists():
                delay = get_config_delay(config_file)
                if delay != 'N/A':
                    df = parse_cma_log(log_file)
                    if not df.empty:
                        df['delay'] = delay
                        all_results.append(df)

    if not all_results:
        print(f"Error: No valid CMA-ES log files and config.json pairs found in subdirectories of {results_path}")
        return

    full_df = pd.concat(all_results, ignore_index=True)

    # Generate plots
    plot_s_curve(full_df, output_path / 's_curve_summary.png')
    plot_combined_progress(full_df, output_path / 'combined_optimization_progress.png')

if __name__ == "__main__":
    main() 