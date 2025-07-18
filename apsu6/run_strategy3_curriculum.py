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

# Globals for multiprocessing
harness = None

def setup_logging(results_dir: Path):
    log_path = results_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

def init_worker(config):
    global harness
    worker_seed = config['seed'] + os.getpid()
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    harness = ExperimentHarness(config)

def evaluate_fitness_parallel(solution_vector, readout_mode, config):
    global harness
    s_score, diagnostics = harness.evaluate_fitness(
        solution_vector, 
        readout_mode=readout_mode,
        num_avg=config['evaluation']['num_avg']
    )
    # For now, we use a simple fitness function. Reward shaping can be added later.
    return -s_score, s_score, diagnostics

def main():
    parser = argparse.ArgumentParser(description="Run Strategy 3: Curriculum Learning with Annealing Readout.")
    parser.add_argument('--config', type=str, required=True, help="Path to a JSON config file.")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel worker processes.")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    with open(args.config, 'r') as f:
        config = json.load(f)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"apsu6/results/strategy3_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir)

    logging.info("--- Starting Strategy 3: Curriculum Learning ---")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    temp_harness = ExperimentHarness(config)
    num_params = temp_harness.get_solution_dimension()
    del temp_harness

    # --- Phase 1: Post-Hoc Readout (Easy Problem) ---
    logging.info(f"--- Phase 1: Optimizing with Post-Hoc Readout ---")
    es_phase1 = cma.CMAEvolutionStrategy(num_params * [0], 0.5, {'popsize': config['optimizer']['population_size'], 'seed': config['seed']})
    
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
        for gen in range(config['optimizer']['generations_phase1']):
            solutions = es_phase1.ask()
            eval_func = partial(evaluate_fitness_parallel, readout_mode='post_hoc', config=config)
            results = pool.map(eval_func, solutions)
            fitnesses, s_scores, _ = zip(*results)
            es_phase1.tell(solutions, list(fitnesses))
            logging.info(f"[Phase 1, Gen {gen+1}] Best S-Score: {max(s_scores):.4f}")

    best_solution_phase1 = es_phase1.result.xbest
    logging.info("--- Phase 1 Complete ---")
    np.save(results_dir / "best_solution_phase1.npy", best_solution_phase1)

    # --- Phase 2: End-to-End Readout (Hard Problem) ---
    logging.info(f"--- Phase 2: Fine-tuning with End-to-End Readout ---")
    # Warm-start the optimizer with the solution from Phase 1
    es_phase2 = cma.CMAEvolutionStrategy(best_solution_phase1, 0.2, {'popsize': config['optimizer']['population_size'], 'seed': config['seed'] + 1})

    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
        for gen in range(config['optimizer']['generations_phase2']):
            solutions = es_phase2.ask()
            eval_func = partial(evaluate_fitness_parallel, readout_mode='end_to_end', config=config)
            results = pool.map(eval_func, solutions)
            fitnesses, s_scores, _ = zip(*results)
            es_phase2.tell(solutions, list(fitnesses))
            logging.info(f"[Phase 2, Gen {gen+1}] Best S-Score: {max(s_scores):.4f}")
    
    best_solution_final = es_phase2.result.xbest
    logging.info("--- Strategy 3 Complete ---")
    np.save(results_dir / "best_solution_final.npy", best_solution_final)
    
    # Final analysis of the best solution
    logging.info("--- Final Analysis ---")
    harness = ExperimentHarness(config)
    _, final_diagnostics = harness.evaluate_fitness(best_solution_final, readout_mode='end_to_end')
    logging.info(json.dumps(final_diagnostics, indent=2))

if __name__ == "__main__":
    main() 