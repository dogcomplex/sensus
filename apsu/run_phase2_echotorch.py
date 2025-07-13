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
import pickle

# Import the EchoTorch-specific components
from .non_local_coordinator import NonLocalCoordinator
from .non_local_coordinator_small import NonLocalCoordinator_small
from .experiment_echotorch import run_chsh_trial_echotorch
from .classical_system_echotorch import ClassicalSystemEchoTorch

# --- Configuration ---
ESN_DIMENSION = 100
HIDDEN_DIM = 256
MAX_GENERATIONS = 10
# Let's use a smaller population for faster iteration during performance tuning
POPULATION_SIZE = mp.cpu_count() if mp.cpu_count() <= 12 else 12

# --- Globals for Multiprocessing ---
# These will be initialized once per worker process
system = None
controller = None


def init_worker(device, is_linear, use_small_controller):
    """Initializes the simulation environment for each worker process."""
    global system, controller
    # Create a single, persistent ClassicalSystem instance for this worker
    system = ClassicalSystemEchoTorch(N=ESN_DIMENSION, device=device)
    
    # Choose the controller class based on the flag
    ControllerClass = NonLocalCoordinator_small if use_small_controller else NonLocalCoordinator
    
    # Create a single controller instance to be updated with weights
    controller = ControllerClass(
        esn_dimension=ESN_DIMENSION,
        hidden_dim=HIDDEN_DIM, 
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

def evaluate_fitness(weights, device, delay):
    """
    Fitness function for the CMA-ES optimizer.
    It now uses the persistent system and controller from the worker's global scope.
    """
    global system, controller
    set_weights(controller, weights)
    
    if device.type == 'cuda':
        controller.half()

    s_score = run_chsh_trial_echotorch(controller, system, seed=42, device=device, delay=delay)
    
    return -s_score

def main(delay, is_linear_controller, use_small_controller):
    """
    Main function using an efficient multiprocessing strategy.
    """
    controller_type = "Linear" if is_linear_controller else "Non-Linear"
    controller_size = "Small" if use_small_controller else "Large"
    print(f"--- Starting Project Apsu (EchoTorch): Phase 2 (Optimized) ---")
    print(f"Controller Type: {controller_type} ({controller_size})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Optimizer State Caching ---
    # Create a directory to store optimizer states
    save_dir = "apsu/cma_es_states"
    os.makedirs(save_dir, exist_ok=True)
    optimizer_state_file = os.path.join(save_dir, f"cma_es_{controller_size.lower()}_{controller_type.lower()}_d_{delay}.pkl")
    
    # We only need to get the number of parameters on the main process
    # The actual controller instances will be created in the workers
    ControllerClass = NonLocalCoordinator_small if use_small_controller else NonLocalCoordinator
    temp_controller = ControllerClass(
        esn_dimension=ESN_DIMENSION, hidden_dim=HIDDEN_DIM, is_linear=is_linear_controller
    )
    n_params = sum(p.numel() for p in temp_controller.parameters())
    del temp_controller
    print(f"Optimizing a {controller_size.upper()} {controller_type.upper()} controller with {n_params} parameters.")

    cma_options = {
        'popsize': POPULATION_SIZE,
        'CMA_diagonal': True,
        'seed': 42
    }
    
    es = None
    if os.path.exists(optimizer_state_file):
        with open(optimizer_state_file, 'rb') as f:
            es = pickle.load(f)
        print(f"Resumed optimizer state from {optimizer_state_file}")
        if es.N != n_params:
            print("Warning: Optimizer dimension mismatch! Starting new optimization.")
            es = None

    if es is None:
        es = cma.CMAEvolutionStrategy(n_params * [0], 0.5, cma_options)
        print("Starting new optimization run.")

    print(f"Running CMA-ES for {MAX_GENERATIONS} generations with population size {POPULATION_SIZE}.")

    best_s_scores = []
    
    # Use the 'spawn' context for CUDA safety
    # Initialize each worker process with the init_worker function
    with mp.get_context("spawn").Pool(initializer=init_worker, initargs=(device, is_linear_controller, use_small_controller)) as pool:
        with tqdm(range(MAX_GENERATIONS), desc="CMA-ES Generations") as pbar:
            for gen in pbar:
                solutions = es.ask()
                
                # Use partial to pre-fill the non-changing arguments
                partial_eval = partial(evaluate_fitness, device=device, delay=delay)
                
                # The pool will now efficiently reuse the workers and their environments
                fitnesses = pool.map(partial_eval, solutions)
                
                es.tell(solutions, fitnesses)

                # Save the full optimizer state after each generation
                with open(optimizer_state_file, 'wb') as f:
                    pickle.dump(es, f)
                
                best_s_in_gen = -es.result.fbest
                best_s_scores.append(best_s_in_gen)
                
                pbar.set_postfix({"best_S": f"{best_s_in_gen:.4f}"})

    print("\nOptimization complete.")
    print(f"Best S-score found: {-es.result.fbest:.4f}")

    # Plotting and saving results...
    controller_type_str = f"{controller_size} {controller_type}"

    # Check against Success Gate C1 from spec §1.3
    # Note: Individual plot generation is now part of main()
    individual_plot_path = f"apsu/phase2_{controller_size.lower()}_{controller_type.lower()}_d_{delay}_results.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_s_scores, marker='o')
    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.axhline(2.828, color='g', linestyle='--', label="Tsirelson's Bound (S≈2.828)")
    
    # Add dynamic y-axis scaling for better visualization
    if best_s_scores:
        min_s = min(best_s_scores)
        max_s = max(best_s_scores)
        # Add a bit of padding to the top and bottom
        y_margin = (max_s - min_s) * 0.15 if (max_s - min_s) > 0.01 else 0.1
        plot_min = max(0, min_s - y_margin)
        # Ensure the classical bound is always visible, but don't zoom out too far
        plot_max = max(max_s + y_margin, 2.2) 
        plt.ylim(plot_min, plot_max)

    plt.title(f"Best S-Score vs. Generation ({controller_type_str} Controller, d={delay})")
    plt.xlabel("Generation")
    plt.ylabel("Best S-Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(individual_plot_path)
    print(f"Results plot saved to {individual_plot_path}")
    plt.close()
    
    if max(best_s_scores) > 2.02:
        print(f"\nSuccess Gate Passed: {controller_type_str.upper()} controller VIOLATED the classical bound!")
    else:
        print(f"\nSuccess Gate FAILED: {controller_type_str.upper()} controller was unable to violate the classical bound.")
        
    print(f"--- Phase 2 (EchoTorch, {controller_type_str}) Complete for d={delay} ---")
    # Return both the final best score and the full history for later plotting
    return -es.result.fbest, best_s_scores

if __name__ == "__main__":
    # --- The S(R) Curve Generation ---
    # As per spec §2.3 and §5.4, we test a set of controller delays.
    # We will skip d=0.5 for now as it requires special handling.
    delay_values = [1, 2, 3, 5, 8, 13]
    
    # Store the best S-score found for each delay.
    final_s_scores = {}
    # Store the full optimization history for each delay.
    optimization_histories = {}

    # --- Control Flag ---
    # Set to True to run with a Linear controller, False for Non-Linear
    use_linear_controller = True
    # --- New Control Flag ---
    # Set to True to use the smaller controller architecture
    use_small_controller = True
    
    controller_type = "Linear" if use_linear_controller else "Non-Linear"
    controller_size = "Small" if use_small_controller else "Large"

    for d in delay_values:
        print(f"--- Running experiment with controller delay d={d} ---")
        # For the main experiment, we use the non-linear controller.
        best_s, history = main(delay=d, is_linear_controller=use_linear_controller, use_small_controller=use_small_controller)
        final_s_scores[d] = best_s
        optimization_histories[d] = history
        print(f"Best S-score for d={d}: {best_s:.4f}")

    print("\n\n--- Full S(R) Curve Experiment Complete ---")
    print("Final Results:")
    for d, s in final_s_scores.items():
        print(f"  d = {d:<2} | S = {s:.4f}")

    # --- Generate the Final Plots ---
    
    # 1. Combined Optimization History Plot
    plt.figure(figsize=(12, 7))
    for d, history in sorted(optimization_histories.items()):
        plt.plot(history, marker='o', linestyle='-', label=f'd = {d}')

    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.title(f"Optimization History for All Controller Delays ({controller_size} {controller_type})", fontsize=16)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Best S-Score in Generation", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    history_plot_path = f"apsu/optimization_history_{controller_size.lower()}_{controller_type.lower()}_echotorch.png"
    plt.savefig(history_plot_path)
    print(f"\nCombined optimization history plot saved to {history_plot_path}")
    plt.close()


    # 2. Final S(R) Curve Plot
    delays = sorted(final_s_scores.keys())
    scores = [final_s_scores[d] for d in delays]

    plt.figure(figsize=(12, 7))
    plt.plot(delays, scores, marker='o', linestyle='-', color='b')
    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.axhline(2.828, color='g', linestyle='--', label="Tsirelson's Bound (S≈2.828)")
    
    plt.title(f"S(R) Curve: Best S-Score vs. Controller Delay ({controller_size} {controller_type})", fontsize=16)
    plt.xlabel("Controller Delay (d) [lower is faster]", fontsize=12)
    plt.ylabel("Best Attainable S-Score", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xscale('log') # Delay is often best viewed on a log scale
    
    # Add annotations
    for d, s in zip(delays, scores):
        plt.text(d, s, f' {s:.3f}', va='bottom' if s > 2.1 else 'top')

    final_plot_path = f"apsu/S_vs_R_curve_{controller_size.lower()}_{controller_type.lower()}_echotorch.png"
    plt.savefig(final_plot_path)
    print(f"\nFinal S(R) curve plot saved to {final_plot_path}")
    plt.close() 