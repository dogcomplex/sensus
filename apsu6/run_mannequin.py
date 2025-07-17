import torch
import cma
import numpy as np
import json
import time
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import reservoirpy as rpy

from apsu6.harness import ExperimentHarness

def setup_logging(results_dir: Path):
    """Configures logging to file and console."""
    log_path = results_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def get_config(args):
    """Creates the main configuration dictionary."""
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
        
        "substrate_params": {
            "N_A": 50, "N_B": 50,
            "sr_A": 0.7, "sr_B": 0.7,
            "lr_A": 0.7, "lr_B": 0.7,
            "noise_A": 0.0, "noise_B": 0.0,
            "seed_A": args.seed + 10, "seed_B": args.seed + 11
        },
        
        "controller_params": {
            "protocol": "Mannequin",
            "N_A": 50, "N_B": 50,
            "K_controller": 32,
            "R_speed": 1.0 / args.delay if args.delay > 0 else float('inf'),
            "signaling_bits": 0, # Not used in Mannequin
            "internal_cell_config": {
                "enabled": True, # ENABLED FOR R > 1
                "type": "gru", 
                "hidden_size": 32
            }
        },
        
        "optimizer": {
            "generations": 100,
            "population_size": 12,
            "teacher_lambda": 1.0 # Weight for the PR-Box teacher loss
        }
    }
    return config

def main():
    rpy.verbosity(0)
    parser = argparse.ArgumentParser(description="Run a Protocol M (Mannequin) CHSH experiment.")
    parser.add_argument('--controller-units', type=int, default=32, help="Number of controller units (K_controller).")
    parser.add_argument('--seed', type=int, default=42, help="Main random seed.")
    parser.add_argument('--delay', type=float, default=0.1, help="Controller delay 'd' for Speed Ratio R=1/d.")
    parser.add_argument('--teacher', action='store_true', help="Use the PR-Box teacher to guide the optimizer.")
    args = parser.parse_args()

    config = get_config(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"apsu6/results/mannequin_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir)

    logging.info("--- Starting Protocol M (Mannequin) Experiment ---")
    logging.info(f"Results will be saved to: {results_dir}")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    logging.info("Configuration: \n" + json.dumps(config, indent=2))

    harness = ExperimentHarness(config)
    
    temp_controller = harness.controller
    num_params = sum(p.numel() for p in temp_controller.parameters())
    logging.info(f"Total parameters to optimize: {num_params}")

    es = cma.CMAEvolutionStrategy(num_params * [0], 0.5, {
        'popsize': config['optimizer']['population_size'],
        'seed': config['seed']
    })

    history = {'s_scores': [], 'best_s': -4.0}
    pbar = tqdm(total=config['optimizer']['generations'], desc="Optimizing (Protocol M)")

    for generation in range(config['optimizer']['generations']):
        solutions_params = es.ask()
        fitness_scores = []

        for params in solutions_params:
            controller_weights = {}
            start_idx = 0
            for name, param in temp_controller.named_parameters():
                n_params = param.numel()
                p_slice = torch.from_numpy(params[start_idx : start_idx + n_params]).view(param.shape).float()
                controller_weights[name] = p_slice
                start_idx += n_params
            
            s_score, diagnostics = harness.evaluate_fitness(controller_weights)
            
            # Combine S-score with teacher loss for the final fitness
            teacher_loss = diagnostics.get("pr_box_teacher_loss", 0.0)
            teacher_lambda = config['optimizer'].get('teacher_lambda', 0.0)
            
            fitness = -s_score + (teacher_lambda * teacher_loss)
            fitness_scores.append(fitness)

        es.tell(solutions_params, fitness_scores)
        
        # We still want to track the pure S-score for progress reporting
        best_solution_in_gen_params = solutions_params[np.argmin(fitness_scores)]
        
        # Recreate weights dict for the best individual to re-evaluate for diagnostics
        best_weights_for_eval = {}
        start_idx = 0
        for name, param in temp_controller.named_parameters():
            num_params = param.numel()
            p_slice = torch.from_numpy(best_solution_in_gen_params[start_idx : start_idx + num_params]).view(param.shape).float()
            best_weights_for_eval[name] = p_slice
            start_idx += num_params

        s_score_of_best, diags_of_best = harness.evaluate_fitness(best_weights_for_eval)

        best_s_in_gen = s_score_of_best
        history['s_scores'].append(best_s_in_gen)
        
        if best_s_in_gen > history['best_s']:
            history['best_s'] = best_s_in_gen
            best_weights_params = es.result.xbest
            
            best_weights_dict = {}
            start_idx = 0
            for name, param in temp_controller.named_parameters():
                n_params = param.numel()
                p_slice = torch.from_numpy(best_weights_params[start_idx : start_idx + n_params]).view(param.shape).float()
                best_weights_dict[name] = p_slice
                start_idx += n_params

            torch.save(best_weights_dict, results_dir / "best_controller.pth")

        pbar.set_postfix({"S": f"{best_s_in_gen:.4f}", "Best S": f"{history['best_s']:.4f}", "Teacher Loss": f"{diags_of_best.get('pr_box_teacher_loss', 0.0):.4f}"})
        pbar.update(1)

    pbar.close()

    logging.info("--- Experiment Complete ---")
    logging.info(f"Final Best S-Score: {history['best_s']:.4f}")
    with open(results_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main() 