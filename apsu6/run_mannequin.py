import torch
import cma
import numpy as np
import json
import time
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial

from apsu6.harness import ExperimentHarness

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

def get_config(args):
    """Creates the main configuration dictionary."""
    # This function remains largely the same
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
                "enabled": True, "type": "gru",
                "hidden_size": args.controller_units
            }
        },
        "optimizer": {"generations": 100, "population_size": 12},
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
    global harness, temp_controller
    # Each worker gets its own harness and controller instance.
    # This is crucial to avoid CUDA errors with forked processes.
    harness = ExperimentHarness(config)
    temp_controller = harness.controller

def evaluate_fitness_parallel(params, current_lambda, current_noise, config):
    """
    Fitness function to be executed by each worker.
    It uses the persistent harness from the worker's global scope.
    """
    global harness, temp_controller
    
    # Reconstruct the weights dictionary
    controller_weights = {}
    start_idx = 0
    for name, param in temp_controller.named_parameters():
        n_params = param.numel()
        p_slice = torch.from_numpy(params[start_idx : start_idx + n_params]).view(param.shape).float().to(harness.device)
        controller_weights[name] = p_slice
        start_idx += n_params

    s_score, diagnostics = harness.evaluate_fitness(
        controller_weights, 
        sensor_noise_std=current_noise,
        num_avg=config['evaluation']['num_avg']
    )
    
    teacher_loss = diagnostics.get("pr_box_teacher_loss", 0.0)
    fitness = -s_score + (current_lambda * teacher_loss)
    return fitness, s_score, diagnostics

def main():
    parser = argparse.ArgumentParser(description="Run a Protocol M (Mannequin) CHSH experiment.")
    parser.add_argument('--controller-units', type=int, default=32, help="Controller units (K).")
    parser.add_argument('--seed', type=int, default=42, help="Main random seed.")
    parser.add_argument('--delay', type=float, default=0.1, help="Controller delay 'd' (R=1/d).")
    parser.add_argument('--teacher', action='store_true', help="Use PR-Box teacher.")
    parser.add_argument('--curriculum', action='store_true', help="Enable curriculum learning.")
    parser.add_argument('--device', type=str, default='cuda', help="Device ('cuda' or 'cpu').")
    parser.add_argument('--num-avg', type=int, default=10, help="Runs to average per evaluation.")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel worker processes.")
    args = parser.parse_args()

    # --- Enforce reproducibility ---
    torch.use_deterministic_algorithms(True)
    # It's important to set the start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    config = get_config(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"apsu6/results/mannequin_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir, is_main_process=True)

    logging.info(f"--- Starting Protocol M Experiment (Parallel) on device: {args.device} ---")
    
    # Log config and other details as before
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    logging.info("Configuration: \n" + json.dumps(config, indent=2))

    temp_harness = ExperimentHarness(config)
    num_params = sum(p.numel() for p in temp_harness.controller.parameters())
    del temp_harness
    logging.info(f"Total parameters to optimize: {num_params}")

    es = cma.CMAEvolutionStrategy(num_params * [0], 0.5, {
        'popsize': config['optimizer']['population_size'],
        'seed': config['seed'], 'CMA_diagonal': True
    })

    history = {'s_scores': [], 'best_s': -4.0}
    pbar = tqdm(total=config['optimizer']['generations'], desc="Optimizing (Parallel)")

    # Create the multiprocessing pool
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
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

            solutions_params = es.ask()
            
            # Map the evaluation function to the pool of workers
            eval_func = partial(evaluate_fitness_parallel, 
                                current_lambda=current_lambda, 
                                current_noise=current_noise, 
                                config=config)
            
            results = pool.map(eval_func, solutions_params)
            
            fitness_scores, s_scores, diagnostics_list = zip(*results)

            es.tell(solutions_params, list(fitness_scores))
            
            best_idx_in_gen = np.argmin(fitness_scores)
            best_s_in_gen = s_scores[best_idx_in_gen]
            best_diags_in_gen = diagnostics_list[best_idx_in_gen]

            history['s_scores'].append(best_s_in_gen)
            
            if best_s_in_gen > history['best_s']:
                history['best_s'] = best_s_in_gen
                best_weights_params = es.result.xbest
                
                # Save the best model's raw parameter vector.
                # This is more robust than reconstructing the state_dict in the main loop.
                save_path = results_dir / "best_controller_weights.npy"
                np.save(save_path, best_weights_params)

            pbar.set_postfix({
                "S": f"{best_s_in_gen:.4f}", "Best S": f"{history['best_s']:.4f}",
                "Loss": f"{best_diags_in_gen.get('pr_box_teacher_loss', 0.0):.4f}",
                "λ": f"{current_lambda:.2f}", "σ": f"{current_noise:.3f}"
            })
            pbar.update(1)

    pbar.close()
    logging.info("--- Experiment Complete ---")
    logging.info(f"Final Best S-Score: {history['best_s']:.4f}")
    # Save history as before
    with open(results_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main() 