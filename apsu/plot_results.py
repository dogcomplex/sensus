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

def get_controller_units(config_path):
    """Reads a config file and extracts the reservoir controller's units."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Path to the units for the ReservoirController
            units = config.get('controller', {}).get('config', {}).get('units', 'N/A')
            if units != 'N/A':
                return int(units)
            return 'N/A'
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

def plot_reservoir_sweep(results_df, output_path):
    """Plots the final S-score vs. Reservoir Controller Units."""
    final_scores = results_df.loc[results_df.groupby('units')['best_fitness'].idxmax()]
    final_scores = final_scores.sort_values('units')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.lineplot(data=final_scores, x='units', y='best_fitness', marker='o', ax=ax, markersize=10, lw=2.5)
    ax.axhline(2.0, color='red', linestyle='--', label='Classical Bound (S=2)')
    ax.axhline(2 * np.sqrt(2), color='green', linestyle='--', label="Tsirelson's Bound (S≈2.828)")
    
    ax.set_title('Best CHSH Score vs. Reservoir Controller Size', fontsize=18)
    ax.set_xlabel('Controller Reservoir Units', fontsize=14)
    ax.set_ylabel('Best Found S-Score', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    
    for _, row in final_scores.iterrows():
        ax.text(row['units'], row['best_fitness'], f" {row['best_fitness']:.4f}", ha='left', va='bottom')
        
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Reservoir sweep plot saved to {output_path}")

def plot_high_res_sweep(results_df, output_path):
    """Plots a bar chart for the high-resolution reservoir sweep."""
    final_scores = results_df.loc[results_df.groupby('units')['best_fitness'].idxmax()]
    final_scores = final_scores.sort_values('units')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # Create a color list: green for success (S>2), blue for failure
    colors = ['#2ca02c' if s > 2.0 else '#1f77b4' for s in final_scores['best_fitness']]

    sns.barplot(data=final_scores, x='units', y='best_fitness', ax=ax, palette=colors, width=0.8)

    ax.axhline(2.0, color='red', linestyle='--', label='Classical Bound (S=2)')
    ax.axhline(2 * np.sqrt(2), color='purple', linestyle='--', label="Tsirelson's Bound (S≈2.828)")

    ax.set_title('High-Resolution Sweep: Best CHSH Score vs. Controller Size', fontsize=20)
    ax.set_xlabel('Controller Reservoir Units', fontsize=16)
    ax.set_ylabel('Best Found S-Score', fontsize=16)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, which='both', axis='y', linestyle='--')

    # Annotate bars
    for i, p in enumerate(ax.patches):
        score = final_scores['best_fitness'].iloc[i]
        ax.annotate(f'{score:.3f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"High-resolution sweep plot saved to {output_path}")

def plot_goldilocks_sweep(results_df, output_path):
    """Plots a detailed line graph for the Goldilocks sweep, highlighting the best performer."""
    final_scores = results_df.loc[results_df.groupby('units')['best_fitness'].idxmax()]
    final_scores = final_scores.sort_values('units')
    
    best_performer = final_scores.loc[final_scores['best_fitness'].idxmax()]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the main line
    sns.lineplot(data=final_scores, x='units', y='best_fitness', marker='o', ax=ax, markersize=6, lw=2, label="Best S-Score per Controller Size")
    
    # Add critical lines
    ax.axhline(2.0, color='red', linestyle='--', label='Classical Bound (S=2)')
    ax.axhline(2 * np.sqrt(2), color='green', linestyle='--', label="Tsirelson's Bound (S≈2.828)")

    # Highlight the best point
    ax.scatter(best_performer['units'], best_performer['best_fitness'], color='gold', s=200, zorder=5, marker='*', edgecolors='black', label=f"Peak Performance (S={best_performer['best_fitness']:.4f} at {best_performer['units']} units)")

    ax.set_title('Goldilocks Sweep: Best CHSH Score vs. Controller Size (1-100 Units)', fontsize=22)
    ax.set_xlabel('Controller Reservoir Units', fontsize=16)
    ax.set_ylabel('Best Found S-Score', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--')
    
    # Set x-axis ticks to be more readable for the 1-100 range
    ax.set_xticks(np.arange(0, 101, 5))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Goldilocks sweep plot saved to {output_path}")


def plot_combined_progress(results_df, output_path, group_by='delay'):
    """Plots all optimization progresses on a single graph."""
    # Calculate the cumulative max fitness for each group
    results_df['cumulative_best'] = results_df.groupby(group_by)['best_fitness'].cummax()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    sns.lineplot(data=results_df, x='generation', y='cumulative_best', hue=group_by, palette='viridis', lw=2, ax=ax)
    
    ax.set_title('Cumulative Best Fitness vs. Generation', fontsize=18)
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Best S-Score Found', fontsize=14)
    ax.legend(title=group_by.capitalize())
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
        '--mode',
        type=str,
        default='s_curve',
        choices=['s_curve', 'reservoir_sweep', 'high_res_sweep', 'goldilocks_sweep'],
        help='The type of experiment to plot.'
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
    
    # Logic to find the parameter of interest based on the mode
    if args.mode == 's_curve':
        param_getter = get_config_delay
        param_name = 'delay'
    elif args.mode == 'reservoir_sweep':
        param_getter = get_controller_units
        param_name = 'units'
    elif args.mode == 'high_res_sweep':
        param_getter = get_controller_units
        param_name = 'units'
    elif args.mode == 'goldilocks_sweep':
        param_getter = get_controller_units
        param_name = 'units'
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        return

    # Use rglob to find all config.json files recursively, which is robust
    # to the extra directory layer created by the harness.
    config_files = list(results_path.rglob('config.json'))

    if not config_files:
        print(f"Error: No config.json files found in subdirectories of {results_path}")
        return

    for config_path in config_files:
        exp_dir = config_path.parent
        param_val = param_getter(config_path)
        if param_val == 'N/A':
            continue

        # Look for CMA logs in the config file's directory AND its parent.
        # This is robust to the harness creating an extra timestamped folder
        # while the CMA logger logs to the parent.
        log_files = list(exp_dir.glob('cma_*.dat'))
        if not log_files:
            log_files = list(exp_dir.parent.glob('cma_*.dat'))

        if not log_files:
            print(f"Warning: No cma_*.dat log file found for config {config_path}")
            continue
        
        # We assume the most recent .dat file is the one of interest if there are multiple.
        log_path = max(log_files, key=os.path.getmtime)
        
        df = parse_cma_log(log_path)
        if not df.empty:
            df[param_name] = param_val
            all_results.append(df)

    if not all_results:
        print(f"Error: No valid CMA-ES log files and config.json pairs found in subdirectories of {results_path}")
        return

    results_df = pd.concat(all_results)
    
    # Generate the main plot based on the mode
    if args.mode == 's_curve':
        plot_s_curve(results_df, output_path / 's_curve_summary.png')
    elif args.mode == 'reservoir_sweep':
        plot_reservoir_sweep(results_df, output_path / 'reservoir_sweep_summary.png')
    elif args.mode == 'high_res_sweep':
        plot_high_res_sweep(results_df, output_path / 'high_res_sweep_summary.png')
    elif args.mode == 'goldilocks_sweep':
        plot_goldilocks_sweep(results_df, output_path / 'goldilocks_sweep_summary.png')

    # Also generate the combined progress plot
    progress_plot_path = output_path / f'{args.mode}_progress_over_time.png'
    plot_combined_progress(results_df, progress_plot_path, group_by=param_name)

    # Print a text summary of the final scores
    print("\n--- Final Scores Summary ---")
    final_scores = results_df.loc[results_df.groupby(param_name)['best_fitness'].idxmax()]
    final_scores = final_scores.sort_values(param_name)
    print(final_scores[[param_name, 'best_fitness']].to_string(index=False))

if __name__ == "__main__":
    main() 