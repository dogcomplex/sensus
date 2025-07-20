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
import gc

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

def evaluate_fitness_parallel(solution_vector, config):
    global harness
    s_score, diagnostics = harness.evaluate_fitness(
        solution_vector
    )
    
    fitness = -s_score # Basic fitness
    
    # Apply reward shaping based on statistical significance
    reward_config = config.get("reward_shaping", {"enabled": False})
    if reward_config.get("enabled", False):
        s_score_lower_bound = diagnostics.get("S_score_ci", (0.0, 0.0))[0]
        if s_score_lower_bound > reward_config.get("tsirelson_bound", 2.828):
            fitness -= reward_config.get("quantum_bonus", 2.0)
        elif s_score_lower_bound > reward_config.get("classical_bound", 2.0):
            fitness -= reward_config.get("classical_bonus", 1.0)
            
    return fitness, s_score, diagnostics

def run_optimization_phase(phase_name, es_instance, generations, pool, config):
    logging.info(f"--- Starting {phase_name} ---")
    history = {'best_s': -4.0, 'best_diagnostics': {}}
    
    pbar = tqdm(total=generations, desc=phase_name)
    for gen in range(generations):
        solutions = es_instance.ask()
        eval_func = partial(evaluate_fitness_parallel, config=config)
        results = pool.map(eval_func, solutions)
        fitnesses, s_scores, diagnostics_list = zip(*results)
        es_instance.tell(solutions, list(fitnesses))
        
        best_idx = np.argmax(s_scores)
        best_s_gen = s_scores[best_idx]
        
        if best_s_gen > history['best_s']:
            history['best_s'] = best_s_gen
            history['best_diagnostics'] = diagnostics_list[best_idx]
            
        pbar.set_postfix({"Best S": f"{history['best_s']:.4f}"})
        pbar.update(1)

        # Aggressively clean up memory after each generation
        del results
        del fitnesses
        del s_scores
        del diagnostics_list
        gc.collect()
        
    pbar.close()
    logging.info(f"--- {phase_name} Complete ---")
    logging.info(f"Best S-Score in phase: {history['best_s']:.4f}")
    return es_instance, history['best_diagnostics']

def main():
    parser = argparse.ArgumentParser(description="Run the Constrained Chaos Strategy.")
    parser.add_argument('--config', type=str, required=True, help="Path to the unified strategy JSON config.")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel worker processes.")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    with open(args.config, 'r') as f:
        config = json.load(f)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"apsu6/results/constrained_chaos_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(results_dir)

    logging.info("--- Starting Constrained Chaos Strategy ---")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # --- Single, Constrained Optimization ---
    # The harness now handles the "constrained chaos" clipping internally
    temp_harness = ExperimentHarness(config)
    num_params = temp_harness.get_solution_dimension()
    del temp_harness
    
    es = cma.CMAEvolutionStrategy(
        num_params * [0], 0.5, 
        {'popsize': config['optimizer']['population_size'], 'seed': config['seed'], 'CMA_diagonal': True}
    )

    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
        es_final, diags_final = run_optimization_phase(
            "Constrained Chaos Run", es, config['optimizer']['generations'],
            pool, config
        )
    
    best_solution_final = es_final.result.xbest
    np.save(results_dir / "best_solution_final.npy", best_solution_final)
    with open(results_dir / "diagnostics_final.json", 'w') as f:
        if 'correlations' in diags_final:
            diags_final['correlations'] = {str(k): v for k, v in diags_final['correlations'].items()}
        json.dump(diags_final, f, indent=2)
    
    logging.info("--- Constrained Chaos Strategy Complete ---")
    logging.info("Final Diagnostics:")
    if 'correlations' in diags_final:
        diags_final['correlations'] = {str(k): v for k, v in diags_final['correlations'].items()}
    logging.info(json.dumps(diags_final, indent=2))

if __name__ == "__main__":
    main() 