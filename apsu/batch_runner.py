import subprocess
import os
import argparse
import sys
import shutil
import json
import time
from pathlib import Path
import random

PYTHON_EXECUTABLE = sys.executable

def generate_s_curve_commands():
    """Generates experiment configurations and commands for the S(d) curve."""
    print("--- Generating S(R) curve experiment configurations... ---")
    base_config_path = Path("apsu/experiments/cma_es/full_config.json")
    if not base_config_path.exists():
        print(f"ERROR: Base configuration '{base_config_path}' not found! Please create it.")
        sys.exit(1)

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Per REQUIREMENTS_v4.md, section 2.3
    d_values = [0.5, 1, 2, 3, 5, 8, 13]
    commands = []
    
    # Always run phase 0 and 1 first as a baseline check
    commands.append(f"{PYTHON_EXECUTABLE} -m apsu.run_phase0")
    
    # Generate Phase 1 full config on the fly if it doesn't exist
    phase1_full_config_path = Path("apsu/experiments/phase1/phase1_full_config.json")
    if not phase1_full_config_path.exists():
        print(f"Generating missing config: {phase1_full_config_path}")
        # This part assumes phase1_fast_config.json exists as a template.
        # A more robust solution might define this config directly in code.
        phase1_fast_config_path = Path("apsu/experiments/phase1/phase1_fast_config.json")
        if not phase1_fast_config_path.exists():
             print(f"ERROR: Template config '{phase1_fast_config_path}' not found! Cannot generate full phase 1 config.")
             sys.exit(1)

        with open(phase1_fast_config_path, 'r') as f:
            phase1_config = json.load(f)
        
        phase1_config["experiment_description"] = "Phase 1 (Full): A rigorous null experiment to validate the CHSH measurement apparatus with a zero controller. Uses a large number of trials for statistical confidence."
        phase1_config["simulation_config"]["washout_time"] = 1000
        phase1_config["simulation_config"]["eval_block_size"] = 1000
        phase1_config["phase1_config"]["num_trials"] = 100
        phase1_config["phase1_config"]["plot_path"] = "apsu/review/phase1/phase1_null_experiment_results_full.png"
        
        with open(phase1_full_config_path, 'w') as f:
            json.dump(phase1_config, f, indent=4)

    commands.append(f"{PYTHON_EXECUTABLE} -m apsu.run_phase1 --config {phase1_full_config_path}")

    output_dir = Path("apsu/experiments/s_curve")
    output_dir.mkdir(exist_ok=True, parents=True)

    for d in d_values:
        config_filename = output_dir / f"d_{str(d).replace('.', '_')}_config.json"

        if not config_filename.exists():
            print(f"Generating config: {config_filename} for d={d}")
            new_config = base_config.copy()
            new_config["chsh_evaluation"]["delay"] = d
            
            # Set generations to 100, overriding any other logic.
            new_config["optimizer"]["config"]["n_generations"] = 100

            with open(config_filename, 'w') as f:
                json.dump(new_config, f, indent=4)
        else:
            print(f"Found existing config: {config_filename}")
        
        commands.append(f"{PYTHON_EXECUTABLE} -m apsu.harness --config={config_filename}")

    print("--- Configuration generation complete. ---")
    return commands

def generate_reservoir_sweep_commands():
    """Generates experiment configurations for a sweep of reservoir controller sizes."""
    print("--- Generating Reservoir Controller sweep configurations... ---")
    base_config_path = Path("apsu/experiments/reservoir/smoke_config.json")
    if not base_config_path.exists():
        print(f"ERROR: Base reservoir config '{base_config_path}' not found!")
        sys.exit(1)

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Sweep from 50 down to 5 units
    controller_sizes = [50, 40, 30, 20, 15, 10, 5]
    commands = []
    
    output_dir = Path("apsu/experiments/reservoir_sweep")
    output_dir.mkdir(exist_ok=True)
    
    base_config["optimizer"]["config"]["n_generations"] = 200 # Full run

    for size in controller_sizes:
        config_filename = output_dir / f"res_controller_{size}_units_config.json"
        
        if not config_filename.exists():
            print(f"Generating config: {config_filename} for controller size={size}")
            new_config = base_config.copy()
            # Set the controller's reservoir size
            new_config["controller"]["config"]["units"] = size
            
            with open(config_filename, 'w') as f:
                json.dump(new_config, f, indent=4)
        
        command = f"{PYTHON_EXECUTABLE} -m apsu.harness --config={config_filename}"
        commands.append(command)
        
    return commands

def generate_high_res_sweep_commands():
    """Generates a focused, high-resolution sweep around the 20-unit 'Goldilocks Zone'."""
    print("--- Generating High-Resolution Reservoir Controller sweep configurations... ---")
    
    # Use our best-performing config as the template
    base_config_path = Path("apsu/experiments/phase2/unit_20_best_controller.json")
    if not base_config_path.exists():
        print(f"ERROR: Base high-res config '{base_config_path}' not found!")
        sys.exit(1)

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Define the focused sweep range around the known best result
    # We now test every integer value from 1 to 26 for a complete picture.
    controller_sizes = list(range(1, 27))
    commands = []
    
    output_dir = Path("apsu/experiments/high_res_sweep")
    output_dir.mkdir(exist_ok=True)

    # Use a solid number of generations for a robust result
    base_config["optimizer"]["config"]["n_generations"] = 100
    base_config["results_dir"] = str(output_dir / "results")

    for size in controller_sizes:
        config_filename = output_dir / f"res_controller_{size}_units_config.json"
        
        # Don't regenerate configs if they already exist
        if not config_filename.exists():
            print(f"Generating config: {config_filename} for controller size={size}")
            new_config = base_config.copy()
            new_config["controller"]["config"]["units"] = size
            
            with open(config_filename, 'w') as f:
                json.dump(new_config, f, indent=4)
        
        command = f"{PYTHON_EXECUTABLE} -m apsu.harness --config={config_filename}"
        commands.append(command)
        
    return commands

def generate_goldilocks_sweep_commands():
    """
    Generates experiment configurations to test the S(R) curve, where R is
    the speed-ratio controlled by the delay parameter 'd'. This is the primary
    experiment described in the project's design documents.
    """
    print("--- Generating S(R) Curve (formerly Goldilocks) configurations... ---")
    
    # Use a standard, robust NLC config as the base
    base_config_path = Path("apsu/experiments/cma_es/full_config.json")
    if not base_config_path.exists():
        print(f"ERROR: Base NLC config '{base_config_path}' not found!")
        sys.exit(1)

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Define the delay values to sweep, per REQUIREMENTS_v4.md, section 2.3
    d_values = [0.5, 1, 2, 3, 5, 8, 13]
    commands = []
    
    # Define a single, fixed controller architecture for the entire sweep.
    # The variable is the delay, not the controller's internal structure.
    base_config["controller"] = {
        "type": "NonLocal",
        "config": {
            "hidden_dim": 32, # A reasonable, fixed size.
            "use_bias": True
        }
    }

    # Generate ONE seed for the entire sweep for scientific validity.
    consistent_chsh_seed = random.randint(0, 2**32 - 1)
    print(f"INFO: Using consistent CHSH seed for all sweep experiments: {consistent_chsh_seed}")
    base_config["chsh_evaluation"]["chsh_seed"] = consistent_chsh_seed
    
    output_dir = Path("apsu/experiments/s_curve_sweep")
    output_dir.mkdir(exist_ok=True)

    base_config["optimizer"]["config"]["n_generations"] = 100
    base_config["optimizer"]["config"]["disable_early_stopping"] = True
    base_config["optimizer"]["config"]["num_workers"] = 2
    
    sweep_results_parent_dir = output_dir / "results"
    base_config["results_dir"] = str(sweep_results_parent_dir)

    for d in d_values:
        d_str = str(d).replace('.', '_')
        config_filename = output_dir / f"d_{d_str}_config.json"
        
        print(f"Generating config: {config_filename} for delay d={d}")
        new_config = base_config.copy()
        new_config["chsh_evaluation"]["delay"] = d
        
        # Ensure each experiment has its own isolated results directory.
        new_config["results_dir"] = str(sweep_results_parent_dir / f"delay_{d_str}")

        with open(config_filename, 'w') as f:
            json.dump(new_config, f, indent=4)
        
        command = f"{PYTHON_EXECUTABLE} -m apsu.harness --config={config_filename}"
        commands.append(command)
        
    return commands

def get_experiment_commands(mode):
    """Gets a list of experiment commands based on the mode."""
    if mode == 'smoke':
        return [
            f"{PYTHON_EXECUTABLE} -m apsu.run_phase0",
            f"{PYTHON_EXECUTABLE} -m apsu.run_phase1 --config apsu/experiments/phase1/phase1_fast_config.json",
            f"{PYTHON_EXECUTABLE} -m apsu.harness --config=apsu/experiments/cma_es/smoke_config.json",
            f"{PYTHON_EXECUTABLE} -m apsu.harness --config=apsu/experiments/sa/smoke_config.json",
            f"{PYTHON_EXECUTABLE} -m apsu.harness --config=apsu/experiments/reservoir/smoke_config.json",
        ]
    elif mode == 's_curve':
        return generate_s_curve_commands()
    elif mode == 'full':
        # A "full" run is now defined as the s_curve run.
        print("INFO: 'full' mode is an alias for 's_curve'. Running the S-curve experiment.")
        return generate_s_curve_commands()
    elif mode == 'reservoir_sweep':
        return generate_reservoir_sweep_commands()
    elif mode == 'high_res_sweep':
        return generate_high_res_sweep_commands()
    elif mode == 'goldilocks_sweep':
        return generate_goldilocks_sweep_commands()
    else:
        print(f"ERROR: Unknown mode '{mode}'")
        sys.exit(1)

def run_command(command_str):
    """Runs a command string, streams its output, and returns success."""
    print(f"--- RUNNING: {command_str} ---")
    try:
        command_parts = command_str.split()
        process = subprocess.Popen(
            command_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
        
        process.wait()
        
        if process.returncode == 0:
            print(f"--- SUCCESS: {command_str} ---")
            return True
        else:
            print(f"--- FAILURE (Exit Code: {process.returncode}): {command_str} ---")
            return False
            
    except Exception as e:
        print(f"--- CRITICAL FAILURE: {command_str} ---")
        print(f"--- Error: {str(e)} ---")
        return False

def clean_project_caches():
    """Aggressively removes caches and old results to ensure a clean run."""
    print("--- Cleaning project caches and old artifacts... ---")
    project_root = Path(__file__).parent.parent
    apsu_dir = project_root / 'apsu'

    # Remove __pycache__ directories
    for path in list(apsu_dir.rglob("__pycache__")):
        if path.is_dir():
            print(f"Removing cache: {path}")
            shutil.rmtree(path, ignore_errors=True)

    print("--- Cache cleaning complete. ---")


def main():
    """
    Main entry point for the batch runner.
    """
    parser = argparse.ArgumentParser(description="Apsu Project Batch Runner")
    parser.add_argument(
        '--mode',
        type=str,
        default='smoke',
        choices=['smoke', 'full', 's_curve', 'reservoir_sweep', 'high_res_sweep', 'goldilocks_sweep'],
        help="The batch of experiments to run: 'smoke' for quick checks, 's_curve' for the main experiment, 'full' is an alias for 's_curve'."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Clears all caches and experiment results for the selected mode before running.'
    )
    args = parser.parse_args()

    print(f"Using Python executable: {PYTHON_EXECUTABLE}")

    # --- Force Clean ---
    if args.force:
        print("--- FORCE flag is active. Cleaning caches and results before run. ---")
        clean_project_caches()
        
        # Also clean the specific results directory for the chosen mode.
        # This now deletes the ENTIRE experiment directory for the mode
        # to ensure stale configs are also removed.
        mode_dir = Path(f"apsu/experiments/{args.mode}")
        if mode_dir.exists() and mode_dir.is_dir():
            print(f"--- FORCE: Removing entire experiment directory: {mode_dir} ---")
            shutil.rmtree(mode_dir)
        else:
            print(f"Warning: Could not find experiment directory for mode '{args.mode}' to clean.")


    commands = get_experiment_commands(args.mode)
    total_experiments = len(commands)
    succeeded_count = 0
    failed_experiments = []

    print(f"Starting batch run in '{args.mode}' mode. Total experiments: {total_experiments}")

    for i, command in enumerate(commands):
        print("\n" + "-"*80)
        print(f"Processing experiment {i+1}/{total_experiments}")
        
        if run_command(command):
            succeeded_count += 1
        else:
            failed_experiments.append(command)
    
    print("\n" + "="*80)
    print("BATCH RUN SUMMARY")
    print("="*80)
    print(f"Total experiments attempted: {total_experiments}")
    print(f"Succeeded: {succeeded_count}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print("\nThe following experiments failed:")
        for cmd in failed_experiments:
            print(f"  - {cmd}")
        print("\nResult: BATCH RUN FAILED")
    else:
        print("\nResult: ALL EXPERIMENTS SUCCEEDED")
    
    print("="*80)

    if failed_experiments:
        sys.exit(1)

if __name__ == "__main__":
    main()
