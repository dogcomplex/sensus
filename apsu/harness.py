import json
import logging
import sys
import os
from datetime import datetime
import torch
from apsu.chsh import evaluate_fitness, _create_controller
from apsu.optimizers.base_optimizer import BaseOptimizer
from apsu.optimizers.cma_optimizer import CMAESOptimizer
from apsu.optimizers.sa_optimizer import SAOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fitness_function_wrapper(eval_args):
    """
    A top-level wrapper to unpack arguments for multiprocessing.Pool.imap.
    This function must be at the top level to be pickleable by multiprocessing.
    """
    weights, config, seed = eval_args
    return evaluate_fitness(weights, config, seed, return_diagnostics=True)

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

def run_experiment(config_path):
    """
    Main function to run an experiment from a config file.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        run_id = f"apsu_experiment_{int(datetime.now().timestamp())}"
        results_dir = config.get("results_dir", "results")
        run_path = os.path.join(results_dir, run_id)
        os.makedirs(run_path, exist_ok=True)
        logging.info(f"Results will be saved in: {run_path}")

        with open(os.path.join(run_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        device = config.get('device', 'cpu')
        temp_controller = _create_controller(config, device)
        n_params = temp_controller.get_n_params()
        logging.info(f"Controller has {n_params} parameters.")
        del temp_controller

        optimizer = get_optimizer(config['optimizer'], n_params, run_path)
        logging.info(f"Using optimizer: {type(optimizer).__name__}")

        optimizer.run(fitness_function_wrapper, config)
        
        logging.info("Optimization finished.")

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred in harness: {e}", exc_info=True)
        sys.exit(1)

def main():
    if len(sys.argv) != 2 or not sys.argv[1].startswith('--config='):
        print("Usage: python -m apsu.harness --config=<path_to_config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1].split('=')[1]
    run_experiment(config_path)

if __name__ == "__main__":
    main() 