import json
import logging
import sys
import os
from datetime import datetime
import torch
from pathlib import Path
import argparse
import uuid

# Force OpenMP to use a single thread to avoid multiprocessing-related memory errors
# that can occur with libraries like scikit-learn on Windows. This must be set
# before any libraries that might use OpenMP are imported.
os.environ['OMP_NUM_THREADS'] = '1'

# Disable reservoirpy progress bars
from reservoirpy.utils import verbosity
verbosity(0)

from apsu.chsh import evaluate_fitness, _create_controller
from apsu.classical_system_reservoirpy import ClassicalSystemReservoirPy
from apsu.optimizers.base_optimizer import BaseOptimizer
from apsu.optimizers.cma_optimizer import CMAESOptimizer
from apsu.optimizers.sa_optimizer import SAOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fitness_function_wrapper(args):
    """
    A top-level wrapper to unpack arguments for multiprocessing.Pool.imap.
    This function must be at the top level to be pickleable by multiprocessing.
    
    This has been refactored to IGNORE the incoming config and instead load a
    definitive config from a known path to bypass multiprocessing serialization bugs.
    """
    individual, run_path_str, payload_id = args
    run_path = Path(run_path_str)
    
    # Load the definitive payload from disk
    payload_path = run_path / f"payload_{payload_id}.json"
    try:
        with open(payload_path, 'r') as f:
            eval_config = json.load(f)
    except FileNotFoundError:
        # Fallback for safety, but this should not be reached.
        logging.error(f"CRITICAL: Could not find payload file {payload_path}")
        return -2.0 # Return a distinctly bad fitness

    return evaluate_fitness(individual, eval_config, return_full_results=True)

def get_optimizer(optimizer_config, dimension, run_path) -> BaseOptimizer:
    """Factory function to get the appropriate optimizer."""
    optimizer_type = optimizer_config.get('type', 'CMAES')
    config = optimizer_config.get('config', {})

    if optimizer_type == 'CMAES':
        return CMAESOptimizer(dimension=dimension, log_dir=run_path, **config)
    elif optimizer_type == 'SA':
        return SAOptimizer(dimension=dimension, log_dir=run_path, **config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def run_experiment(config):
    """
    Main function to run an experiment from a config file.
    """
    try:
        # The config dictionary is now passed directly. No need to open a file.

        # Create a unique directory for this run's results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"apsu_experiment_{timestamp}"
        run_path = Path(config.get("results_dir", "results")) / run_name
        run_path.mkdir(parents=True, exist_ok=True)

        # Save the configuration used for this run
        with open(run_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

        log_file_handler = logging.FileHandler(run_path / 'run.log')
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_file_handler)

        logging.info(f"Results will be saved in: {run_path}")

        device = config.get('device', 'cpu')

        # Create a temporary classical system and controller to determine the number of parameters
        # that the optimizer needs to work with.
        system_config = config.get('classical_system', {})
        controller_config = config.get('controller', {})
        
        # Inject a dummy seed for the temporary system creation if it's not in the config.
        # This is safe because this system is only used to count parameters and is then discarded.
        if 'seed' not in system_config:
            system_config['seed'] = 999

        temp_system = ClassicalSystemReservoirPy(device=device, **system_config)
        temp_controller = _create_controller(controller_config, temp_system)
        
        n_params = temp_controller.get_n_params()
        logging.info(f"Controller has {n_params} parameters.")
        del temp_controller
        del temp_system

        optimizer = get_optimizer(config['optimizer'], n_params, run_path)
        logging.info(f"Using optimizer: {type(optimizer).__name__}")
        
        # Prepare a slimmed-down config dictionary to pass to the evaluation function.
        eval_payload = {
            "chsh_evaluation": config.get("chsh_evaluation", {}),
            "classical_system": config.get("classical_system", {}),
            "controller": config.get("controller", {}),
            "optimizer": config.get("optimizer", {}), # Add optimizer for the base class
            "device": config.get("device", "cpu"),
            "ablate_controller": config.get("ablate_controller", False)
        }

        # --- Scorched Earth Fix ---
        # Serialize the definitive payload to disk with a unique ID.
        # The wrapper function will load this directly, bypassing multiprocessing args.
        payload_id = str(uuid.uuid4())
        payload_path = run_path / f"payload_{payload_id}.json"
        with open(payload_path, 'w') as f:
            json.dump(eval_payload, f, indent=4)
        # --- End Fix ---


        optimizer.run(fitness_function_wrapper, run_path, payload_id)
        
        logging.info("Optimization finished.")

    except FileNotFoundError:
        logging.error(f"Configuration file not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred in harness: {e}", exc_info=True)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run an Apsu experiment.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the experiment configuration JSON file.'
    )
    parser.add_argument(
        '--ablate-controller',
        action='store_true',
        help='If set, the controller runs but its output is ignored (zeroed out) to test for bugs.'
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Pass the ablation flag into the main config dict
    config['ablate_controller'] = args.ablate_controller
    if args.ablate_controller:
        logging.warning("ABLATION MODE: Controller output will be zeroed out.")

    run_experiment(config)


if __name__ == "__main__":
    main() 