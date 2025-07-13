import cma
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid issues with multiprocessing
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from functools import partial
import os
import hashlib
import pickle

# Import the EchoTorch-specific components
from .non_local_coordinator import NonLocalCoordinator
from .experiment_echotorch import run_chsh_trial_echotorch
from .classical_system_echotorch import ClassicalSystemEchoTorch

# --- Configuration ---
ESN_DIMENSION = 100
MAX_GENERATIONS = 100 # Using a proper budget now that we have a working strategy
# Let's cap the concurrency to a safe number to avoid OS-level memory errors
# on Windows when spawning many heavy torch processes.
NUM_WORKERS = 4
# Set a fixed population size that is known to work and is a multiple of workers.
POPULATION_SIZE = 12

# --- Globals for Multiprocessing ---
# These will be initialized once per worker process
system = None
controller = None

def init_worker(device, is_linear, hidden_dim):
    """Initializes the simulation environment for each worker process."""
    global system, controller
    # Create a single, persistent ClassicalSystem instance for this worker
    system = ClassicalSystemEchoTorch(N=ESN_DIMENSION, device=device)
    # Create a single controller instance to be updated with weights
    controller = NonLocalCoordinator(
        esn_dimension=ESN_DIMENSION,
        hidden_dim=hidden_dim, 
        is_linear=is_linear
    ).to(device)

def set_weights(model, weights):
    """Injects a flat vector of weights into a PyTorch model."""
    with torch.no_grad():
        start = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data.copy_(torch.tensor(weights[start:start+num_params]).view_as(param).to(param.device))
            start += num_params

def evaluate_fitness(weights, device, delay, n_avg):
    """
    Fitness function for the CMA-ES optimizer.
    It now uses the persistent system and controller from the worker's global scope.
    It can average the S-score over multiple trials to smooth the landscape.
    """
    global system, controller
    set_weights(controller, weights)
    
    if device.type == 'cuda':
        # Using .half() can speed things up but may affect precision.
        # Let's keep it for now.
        controller.half()

    if n_avg <= 1:
        # Run a single trial
        s_score = run_chsh_trial_echotorch(controller, system, seed=np.random.randint(0, 1e6), device=device, delay=delay)
    else:
        # Run multiple trials and average the scores
        scores = []
        for i in range(n_avg):
            # Use a different seed for each trial in the average
            trial_seed = np.random.randint(0, 1e6)
            s = run_chsh_trial_echotorch(controller, system, seed=trial_seed, device=device, delay=delay)
            scores.append(s)
        s_score = np.mean(scores)

    return -s_score

def main(delay, is_linear_controller, hidden_dim, n_avg, run_id):
    """
    Main function using an efficient multiprocessing strategy.
    Now includes state-saving and resuming capabilities.
    """
    controller_type = "Linear" if is_linear_controller else "Non-Linear"
    print("--- Starting Project Apsu (EchoTorch): Phase 2 (Debug) ---")
    print(f"RUN_ID: {run_id}")
    print(f"Controller: {controller_type} | Hidden Dim: {hidden_dim} | Delay (d): {delay} | Fitness Avg (n_avg): {n_avg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We only need to get the number of parameters on the main process
    temp_controller = NonLocalCoordinator(
        esn_dimension=ESN_DIMENSION, hidden_dim=hidden_dim, is_linear=is_linear_controller
    )
    n_params = sum(p.numel() for p in temp_controller.parameters())
    del temp_controller
    print(f"Optimizing a {controller_type.upper()} controller with {n_params} parameters.")

    cma_options = {
        'popsize': POPULATION_SIZE,
        'CMA_diagonal': True,
        'seed': 42 # Seeding the optimizer itself for reproducibility
    }

    # --- State Caching and Resuming ---
    os.makedirs('apsu/optimizer_cache', exist_ok=True)
    cache_file = f'apsu/optimizer_cache/{run_id}.pkl'
    
    # --- Fallback to old cache file path ---
    controller_size_str = "small" if (hidden_dim is not None and hidden_dim <= 16) else "large"
    controller_type_str = "linear" if is_linear_controller else "non-linear"
    old_cache_dir = 'apsu/cma_es_states'
    # This constructs the old filename based on the user's example
    old_cache_file = os.path.join(old_cache_dir, f"cma_es_{controller_size_str}_{controller_type_str}_d_{delay}.pkl")

    es = None
    # 1. Prioritize loading from the new cache path
    if os.path.exists(cache_file):
        print(f"Found existing optimizer state. Resuming from {cache_file}")
        with open(cache_file, 'rb') as f:
            es = pickle.load(f)
    # 2. Fallback to the old cache path
    elif os.path.exists(old_cache_file):
        print(f"Found OLD optimizer state. Resuming from {old_cache_file}")
        print("This state will be migrated to the new cache format on the next save.")
        with open(old_cache_file, 'rb') as f:
            es = pickle.load(f)
    
    # Verify the loaded dimension matches the current model
    if es and es.N != n_params:
        print(f"Warning: Optimizer dimension mismatch! Expected {n_params}, found {es.N}. Starting new optimization.")
        es = None

    if not es:
        print("No compatible state found. Starting new optimization.")
        es = cma.CMAEvolutionStrategy(n_params * [0], 0.5, cma_options)
    
    print(f"Running CMA-ES for {MAX_GENERATIONS} generations with population size {es.popsize}.")

    best_s_scores = []
    start_gen = es.countiter
    
    with mp.get_context("spawn").Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(device, is_linear_controller, hidden_dim)) as pool:
        with tqdm(range(start_gen, MAX_GENERATIONS), desc="CMA-ES Generations", initial=start_gen, total=MAX_GENERATIONS) as pbar:
            for gen in pbar:
                if es.stop():
                    print("\nOptimizer has signaled to stop. Ending run.")
                    break
                
                solutions = es.ask()
                
                partial_eval = partial(evaluate_fitness, device=device, delay=delay, n_avg=n_avg)
                
                fitnesses = pool.map(partial_eval, solutions)
                
                es.tell(solutions, fitnesses)
                
                # Save state after each successful generation using the standard pickle library
                with open(cache_file, 'wb') as f:
                    pickle.dump(es, f)
                
                best_s_in_gen = -es.result.fbest
                best_s_scores.append(best_s_in_gen)
                
                pbar.set_postfix({"best_S": f"{best_s_in_gen:.4f}"})

    print("\nOptimization complete.")
    best_final_s = -es.result.fbest
    print(f"Best S-score found: {best_final_s:.4f}")

    # --- Plotting individual run results ---
    # We still plot the progress for this specific run
    plt.figure(figsize=(10, 6))
    plt.plot(range(start_gen, start_gen + len(best_s_scores)), best_s_scores, marker='o')
    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.title(f"Best S-Score vs. Generation (RUN_ID: {run_id})")
    plt.xlabel("Generation")
    plt.ylabel("Best S-Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ensure save directory exists
    os.makedirs("apsu/diagnostic_plots", exist_ok=True)
    save_path = f"apsu/diagnostic_plots/run_{run_id}.png"
    plt.savefig(save_path)
    print(f"Individual run plot saved to {save_path}")
    plt.close()
    
    print(f"--- Phase 2 (Debug) Complete for RUN_ID: {run_id} ---")
    return best_final_s, best_s_scores

def generate_run_id(params):
    """Creates a unique, readable ID for a run based on its parameters."""
    # Use a hash for uniqueness but keep params for readability
    param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    hash_obj = hashlib.sha1(param_str.encode())
    return f"{param_str}_{hash_obj.hexdigest()[:6]}"


if __name__ == "__main__":
    # --- Apsu v3.2 - Final S(R) Curve Generation ---
    # Based on diagnostic results, we are proceeding with the winning strategy:
    # - Controller: Small, Non-Linear (hidden_dim=16)
    # - Fitness: Noisy (n_avg=1)
    # This provides the best learning signal.

    # The parameters for our definitive run
    definitive_params = {
        "is_linear": False, 
        "hidden_dim": 16, 
        "n_avg": 1
    }

    # As per spec, we test a set of controller delays.
    delay_values = [1, 2, 3, 5, 8, 13]

    # Store results for final plots
    all_histories = {}
    final_s_scores = {}
    
    print("--- Starting Final S(R) Curve Generation ---")
    for d in delay_values:
        print(f"\n--- Running experiment for d={d} ---")
        
        run_params = {
            "controller": "NL-Small",
            "hd": definitive_params["hidden_dim"],
            "d": d,
            "navg": definitive_params["n_avg"]
        }
        run_id = generate_run_id(run_params)

        best_s, history = main(
            delay=d,
            is_linear_controller=definitive_params["is_linear"],
            hidden_dim=definitive_params["hidden_dim"],
            n_avg=definitive_params["n_avg"],
            run_id=run_id
        )
        final_s_scores[d] = best_s
        all_histories[d] = history

    print("\n\n--- Final S(R) Curve Experiment Complete ---")
    print("Final Results:")
    for d, s in final_s_scores.items():
        print(f"  d = {d:<2} | S = {s:.4f}")

    # --- Generate Final Summary Plots ---
    
    # 1. Combined Optimization History Plot
    plt.figure(figsize=(12, 8))
    for d, history in sorted(all_histories.items()):
        label = f"d = {d}"
        plt.plot(history, marker='o', linestyle='-', label=label, alpha=0.8)

    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.title("S(R) Curve - Optimization Histories", fontsize=16)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Best S-Score in Generation", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    summary_plot_path = "apsu/final_optimization_history.png"
    plt.savefig(summary_plot_path)
    print(f"\nFinal history plot saved to {summary_plot_path}")
    plt.close()

    # 2. Final S(R) Curve Plot
    delays = sorted(final_s_scores.keys())
    scores = [final_s_scores[d] for d in delays]

    plt.figure(figsize=(12, 7))
    plt.plot(delays, scores, marker='o', linestyle='-', color='b')
    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.axhline(2.828, color='g', linestyle='--', label="Tsirelson's Bound (S≈2.828)")
    
    plt.title("S(R) Curve: Best S-Score vs. Controller Delay", fontsize=16)
    plt.xlabel("Controller Delay (d) [lower is faster]", fontsize=12)
    plt.ylabel("Best Attainable S-Score", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xscale('log') 
    
    # Add annotations
    for d, s in zip(delays, scores):
        plt.text(d, s, f' {s:.3f}', va='bottom' if s > 2.1 else 'top')

    final_plot_path = "apsu/final_S_vs_R_curve.png"
    plt.savefig(final_plot_path)
    print(f"\nFinal S(R) curve plot saved to {final_plot_path}")
    plt.close() 