import torch
import cma
import numpy as np
import json
import time
import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial

from apsu6.harness import ExperimentHarness
from apsu6.controller import UniversalController

# --- Globals for Multiprocessing ---
# These will be initialized once per worker process to avoid re-creating
# the harness and controller, which is expensive.
harness = None
temp_controller = None

def setup_logging(results_dir: Path, is_main_process: bool = True):
    """Configures logging. Only the main process logs to console."""
    log_path = results_dir / "run.log"
    handlers = [logging.FileHandler(log_path)]
    if is_main_process:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def get_config_from_args(args):
    """Creates the main configuration dictionary from command-line arguments."""
    # This was the original get_config function, now used as a fallback
    # when no --config file is specified.
    config = {
        "protocol": "Mannequin",
        "seed": args.seed,
        "noise_seed": args.seed + 1,
        "bootstrap_seed": args.seed + 2,
        "controller_delay": args.delay,
        "T_total": 4000,
        "washout_steps": 100,
        "randomness_file": "apsu6/data/chsh_settings.bin",
        "actuation_scale": 1.0,
        "epsilon_ns": 0.02,
        "use_pr_box_teacher": args.teacher,
        "device": args.device,
        "substrate_params": {
            "N_A": 50, "N_B": 50, "sr_A": 0.7, "sr_B": 0.7,
            "lr_A": 0.7, "lr_B": 0.7, "noise_A": 0.0, "noise_B": 0.0,
            "seed_A": args.seed + 10, "seed_B": args.seed + 11
        },
        "controller_params": {
            "protocol": "Mannequin", "N_A": 50, "N_B": 50,
            "K_controller": args.controller_units,
            "R_speed": 1.0 / args.delay if args.delay > 0 else float('inf'),
            "signaling_bits": 0,
            "internal_cell_config": {
                "enabled": True, "type": "gru_layer",
                "hidden_size": args.controller_units,
                "num_layers": max(1, int(round(1.0 / args.delay))) if args.delay > 0 else 1
            }
        },
        "optimizer": {"generations": 100, "population_size": args.population_size},
        "evaluation": {"num_avg": args.num_avg},
        "curriculum": {
            "enabled": args.curriculum,
            "teacher_lambda_start": 5.0, "teacher_lambda_end": 0.5,
            "sensor_noise_start": 0.05, "sensor_noise_end": 0.0
        }
    }
    return config


def init_worker(config):
    """Initializer for each worker process."""
    global harness
    # Diagnostic print
    print(f"[Worker {os.getpid()}] Initializing...")
    # Seed each worker process independently for reproducibility.
    # This is crucial for multiprocessing with CUDA.
    worker_seed = config['seed'] + os.getpid()
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    
    # Each worker gets its own harness instance.
    # The harness no longer creates expensive components on init.
    harness = ExperimentHarness(config)
    # Diagnostic print
    print(f"[Worker {os.getpid()}] Initialization COMPLETE.")


def evaluate_fitness_parallel(solution_vector, current_lambda, current_noise, config):
    """
    Fitness function to be executed by each worker.
    It uses the persistent harness from the worker's global scope.
    """
    global harness
    
    s_score, diagnostics = harness.evaluate_fitness(
        solution_vector, 
        sensor_noise_std=current_noise,
        num_avg=config['evaluation']['num_avg']
    )
    
    teacher_loss = diagnostics.get("pr_box_teacher_loss", 0.0)
    fitness = -s_score + (current_lambda * teacher_loss)
    
    # --- Reward Shaping ---
    # Apply large fitness bonuses for crossing physically significant boundaries.
    # This creates a strong incentive for the optimizer to find these regions.
    reward_config = config.get("reward_shaping", {"enabled": False})
    if reward_config.get("enabled", False):
        s_score_lower_bound = diagnostics.get("S_score_ci", (0.0, 0.0))[0]
        classical_bound = reward_config.get("classical_bound", 2.0)
        tsirelson_bound = reward_config.get("tsirelson_bound", 2.828427)

        if s_score_lower_bound > tsirelson_bound:
            fitness -= reward_config.get("quantum_bonus", 2.0)
        elif s_score_lower_bound > classical_bound:
            fitness -= reward_config.get("classical_bonus", 1.0)

    return fitness, s_score, diagnostics

def main():
    parser = argparse.ArgumentParser(description="Run a Protocol M (Mannequin) CHSH experiment.")
    parser.add_argument('--config', type=str, default=None, help="Path to a JSON config file. If provided, most other arguments are ignored.")
    parser.add_argument('--controller-units', type=int, default=32, help="[No-config fallback] Controller units (K).")
    parser.add_argument('--seed', type=int, default=42, help="Main random seed. Overrides config file if provided.")
    parser.add_argument('--delay', type=float, default=0.1, help="[No-config fallback] Controller delay 'd' (R=1/d).")
    parser.add_argument('--teacher', action='store_true', help="[No-config fallback] Use PR-Box teacher.")
    parser.add_argument('--curriculum', action='store_true', help="[No-config fallback] Enable curriculum learning.")
    parser.add_argument('--device', type=str, default='cuda', help="Device ('cuda' or 'cpu'). Overrides config file if provided.")
    parser.add_argument('--num-avg', type=int, default=12, help="Runs to average per evaluation. Overrides config file if provided.")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument('--half-precision', action='store_true', help="Use float16 for GPU acceleration. Overrides config file if provided.")
    parser.add_argument('--population-size', type=int, default=12, help="Optimizer population size. Overrides config file if provided.")
    args = parser.parse_args()

    # --- Enforce reproducibility ---
    # torch.use_deterministic_algorithms(True) # NOTE: This can cause issues with half-precision and is disabled for performance.
    # It's important to set the start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"--- Loaded configuration from {args.config} ---")
        except FileNotFoundError:
            logging.error(f"Config file not found: {args.config}")
            return
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from config file: {args.config}")
            return
        
        # --- Apply safe, intentional overrides from command line ---
        config['seed'] = args.seed
        config['noise_seed'] = args.seed + 1
        config['bootstrap_seed'] = args.seed + 2
        config['optimizer']['population_size'] = args.population_size
        config['evaluation']['num_avg'] = args.num_avg
        config['device'] = args.device
        config['half_precision'] = args.half_precision

    else:
        # Fallback to old behavior if no --config is provided
        print("--- No config file provided. Building configuration from command-line arguments. ---")
        config = get_config_from_args(args)
        config['half_precision'] = args.half_precision

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"apsu6/results/mannequin_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir, is_main_process=True)

    logging.info(f"--- Starting Protocol M Experiment (Parallel) on device: {args.device} ---")
    
    # Log config and other details as before
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    logging.info("Configuration: \n" + json.dumps(config, indent=2))

    # --- STRATEGY 1 MODIFICATION ---
    # Create a temporary harness on the CPU to get the solution dimension.
    # This is cheap as it doesn't initialize GPU components.
    temp_harness = ExperimentHarness(config)
    num_params = temp_harness.get_solution_dimension()
    del temp_harness
    logging.info(f"Total parameters to optimize: {num_params}")

    # --- CMA-ES Setup ---
    # Ensure the output directory for CMA-ES is inside our results folder
    cma_output_dir = results_dir / "cma_es_out"
    cma_output_dir.mkdir(exist_ok=True)

    es = cma.CMAEvolutionStrategy(num_params * [0], 0.5, {
        'popsize': config['optimizer']['population_size'],
        'seed': config['seed'], 'CMA_diagonal': True,
        'verb_filenameprefix': str(cma_output_dir) + os.sep
    })

    history = {'s_scores': [], 'best_s': -4.0, 'best_diagnostics': {}}
    pbar = tqdm(total=config['optimizer']['generations'], desc="Optimizing (Parallel)")

    # Create the multiprocessing pool
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
        
        # --- Warm-up Run ---
        # Perform a single, small evaluation on each worker to pay the one-time
        # cost of CUDA context creation and kernel compilation. This prevents a 
        # massive stall on the first real evaluation with a large batch size.
        logging.info("Performing warm-up run on all workers...")
        warmup_config = config.copy()
        warmup_config['evaluation']['num_avg'] = 1 # Use a tiny batch size
        # For warmup, we can just pass a zero vector of the correct dimension.
        warmup_params = [np.zeros(num_params)] * args.workers
        
        warmup_eval_func = partial(evaluate_fitness_parallel,
                                   current_lambda=0.0,
                                   current_noise=0.0,
                                   config=warmup_config)
        pool.map(warmup_eval_func, warmup_params)
        logging.info("Warm-up complete. Starting main optimization.")
        
        for generation in range(config['optimizer']['generations']):
            progress = generation / (config['optimizer']['generations'] - 1) if config['optimizer']['generations'] > 1 else 1.0
            
            # Calculate dynamic curriculum parameters
            if config['curriculum']['enabled']:
                lambda_start, lambda_end = config['curriculum']['teacher_lambda_start'], config['curriculum']['teacher_lambda_end']
                current_lambda = lambda_start + progress * (lambda_end - lambda_start)
                noise_start, noise_end = config['curriculum']['sensor_noise_start'], config['curriculum']['sensor_noise_end']
                current_noise = noise_start + progress * (noise_end - noise_start)
            else:
                current_lambda = config['curriculum']['teacher_lambda_end'] if config['use_pr_box_teacher'] else 0.0
                current_noise = 0.0

            solutions = es.ask()
            
            # --- Parallel Evaluation ---
            eval_func = partial(evaluate_fitness_parallel, 
                                current_lambda=current_lambda, 
                                current_noise=current_noise, 
                                config=config)
            
            results = pool.map(eval_func, solutions)
            
            # Unpack results
            fitness_scores, s_scores, diagnostics_list = zip(*results)
            
            es.tell(solutions, list(fitness_scores))
            
            best_idx_in_gen = np.argmin(fitness_scores)
            best_s_in_gen = s_scores[best_idx_in_gen]

            if best_s_in_gen > history['best_s']:
                history['best_s'] = best_s_in_gen
                history['best_diagnostics'] = diagnostics_list[best_idx_in_gen]

            pbar.set_postfix({
                "S": f"{s_scores[best_idx_in_gen]:.4f}", 
                "Best S": f"{history['best_s']:.4f}",
                "Loss": f"{diagnostics_list[best_idx_in_gen].get('pr_box_teacher_loss', 0.0):.4f}",
                "λ": f"{current_lambda:.2f}", "σ": f"{current_noise:.3f}"
            })
            pbar.update(1)

    pbar.close()

    # --- Finalization ---
    logging.info("--- Experiment Complete ---")
    logging.info(f"Final Best S-Score: {history['best_s']:.4f}")
    logging.info("\nDiagnostics for Best Solution Found:")
    logging.info(json.dumps(history['best_diagnostics'], indent=2))
    
    # Save final results
    final_results = {
        "best_s_score": history['best_s'],
        "best_diagnostics": history['best_diagnostics'],
        "config": config,
        "history": {k: v for k, v in history.items() if k not in ['best_diagnostics']}
    }
    with open(results_dir / "results.json", 'w') as f:
        json.dump(final_results, f, indent=2)

    logging.info(f"Saved final results to {results_dir / 'results.json'}")


if __name__ == "__main__":
    main() 