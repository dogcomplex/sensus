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

def run_optimization_phase(phase_name, es_instance, generations, pool, config, readout_mode):
    logging.info(f"--- Starting {phase_name} ---")
    history = {'best_s': -4.0, 'best_diagnostics': {}}
    
    pbar = tqdm(total=generations, desc=phase_name)
    for gen in range(generations):
        solutions = es_instance.ask()
        eval_func = partial(evaluate_fitness_parallel, readout_mode=readout_mode, config=config)
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
        
    pbar.close()
    logging.info(f"--- {phase_name} Complete ---")
    logging.info(f"Best S-Score in phase: {history['best_s']:.4f}")
    return es_instance, history['best_diagnostics']

def main():
    parser = argparse.ArgumentParser(description="Run the Grand Unified Strategy.")
    parser.add_argument('--config', type=str, required=True, help="Path to the unified strategy JSON config.")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument('--resume-from-phase1', type=str, default=None, help="Path to a completed phase 1 results directory to skip to phase 2.")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.resume_from_phase1:
        logging.info(f"--- Resuming from completed Phase 1 at {args.resume_from_phase1} ---")
        results_dir = Path(args.resume_from_phase1)
        # Load the best solution from the provided path
        try:
            best_solution_p1 = np.load(results_dir / "best_solution_phase1.npy")
            with open(results_dir / "diagnostics_phase1.json", 'r') as f:
                diags_p1 = json.load(f)
        except FileNotFoundError:
            logging.error("Error: Could not find 'best_solution_phase1.npy' or 'diagnostics_phase1.json' in the specified directory.")
            return
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"apsu6/results/unified_strategy_{timestamp}")
        results_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(results_dir)

        logging.info("--- Starting Grand Unified Strategy ---")
        with open(results_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # --- Phase 1: Discover the Optimal Universe ---
        phase1_config = config.copy()
        phase1_config['anneal_substrate'] = True
        temp_harness = ExperimentHarness(phase1_config)
        num_params_phase1 = temp_harness.get_solution_dimension()
        del temp_harness
        
        es_phase1 = cma.CMAEvolutionStrategy(
            num_params_phase1 * [0], 0.5, 
            {'popsize': config['optimizer']['population_size'], 'seed': config['seed'], 'CMA_diagonal': True}
        )

        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(phase1_config,)) as pool:
            es_phase1, diags_p1 = run_optimization_phase(
                "Phase 1 (Discover)", es_phase1, config['optimizer']['generations_phase1'],
                pool, phase1_config, 'post_hoc'
            )
        
        best_solution_p1 = es_phase1.result.xbest
        np.save(results_dir / "best_solution_phase1.npy", best_solution_p1)
        # Fix the JSON tuple key error before saving
        if 'correlations' in diags_p1:
            diags_p1['correlations'] = {str(k): v for k, v in diags_p1['correlations'].items()}
        with open(results_dir / "diagnostics_phase1.json", 'w') as f:
            json.dump(diags_p1, f, indent=2)
        # Save the full optimizer state for potential resumption
        with open(results_dir / "cma_es_phase1.pkl", "wb") as f:
            f.write(es_phase1.pickle_dumps())


    # --- Phase 2: Master the Optimal Universe ---
    phase2_config = config.copy()
    phase2_config['anneal_substrate'] = False
    
    # Extract discovered substrate params and lock them in
    temp_harness_p2 = ExperimentHarness(phase2_config)
    controller_dim = sum(p.numel() for p in temp_harness_p2.temp_controller.parameters())
    substrate_hyperparams = best_solution_p1[controller_dim:]
    phase2_config['substrate_params']['sr_A'] = np.clip(substrate_hyperparams[0], 0.7, 1.5)
    phase2_config['substrate_params']['lr_A'] = np.clip(substrate_hyperparams[1], 0.2, 1.0)
    phase2_config['substrate_params']['sr_B'] = np.clip(substrate_hyperparams[2], 0.7, 1.5)
    phase2_config['substrate_params']['lr_B'] = np.clip(substrate_hyperparams[3], 0.2, 1.0)
    
    num_params_phase2 = temp_harness_p2.get_solution_dimension()
    del temp_harness_p2

    # Extract the controller weights from the full Phase 1 solution vector.
    initial_solution_p2 = best_solution_p1[:controller_dim]

    # Warm-start the optimizer with the solution from Phase 1
    es_phase2 = cma.CMAEvolutionStrategy(
        initial_solution_p2, 0.2, 
        {'popsize': config['optimizer']['population_size'], 'seed': config['seed'] + 1, 'CMA_diagonal': True}
    )

    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(phase2_config,)) as pool:
        best_solution_final, diags_final = run_optimization_phase(
            "Phase 2 (Master)", es_phase2, config['optimizer']['generations_phase2'],
            pool, phase2_config, 'end_to_end'
        )

    np.save(results_dir / "best_solution_final.npy", best_solution_final)
    with open(results_dir / "diagnostics_final.json", 'w') as f:
        # Fix the JSON tuple key error before saving
        if 'correlations' in diags_final:
            diags_final['correlations'] = {str(k): v for k, v in diags_final['correlations'].items()}
        json.dump(diags_final, f, indent=2)
    
    logging.info("--- Grand Unified Strategy Complete ---")
    logging.info("Final Diagnostics:")
    # Fix for final printout as well
    if 'correlations' in diags_final:
        diags_final['correlations'] = {str(k): v for k, v in diags_final['correlations'].items()}
    logging.info(json.dumps(diags_final, indent=2))

if __name__ == "__main__":
    main() 