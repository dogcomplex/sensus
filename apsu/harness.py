import json
import logging
import sys
import os
from datetime import datetime
import torch
from pathlib import Path
import argparse

# Force OpenMP to use a single thread to avoid multiprocessing-related memory errors
# that can occur with libraries like scikit-learn on Windows. This must be set
# before any libraries that might use OpenMP are imported.
os.environ['OMP_NUM_THREADS'] = '1'

from apsu.chsh import evaluate_fitness, _create_controller
from apsu.classical_system_echotorch import ClassicalSystemEchoTorch
from apsu.optimizers.base_optimizer import BaseOptimizer
from apsu.optimizers.cma_optimizer import CMAESOptimizer
from apsu.optimizers.sa_optimizer import SAOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fitness_function_wrapper(args):
    """
    A top-level wrapper to unpack arguments for multiprocessing.Pool.imap.
    This function must be at the top level to be pickleable by multiprocessing.
    """
    individual, eval_config = args
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
        temp_system = ClassicalSystemEchoTorch(device=device, **system_config)
        temp_controller = _create_controller(controller_config, temp_system)
        
        n_params = temp_controller.get_n_params()
        logging.info(f"Controller has {n_params} parameters.")
        del temp_controller
        del temp_system

        optimizer = get_optimizer(config['optimizer'], n_params, run_path)
        logging.info(f"Using optimizer: {type(optimizer).__name__}")

        optimizer.run(fitness_function_wrapper, config)
        
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