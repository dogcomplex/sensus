import cma
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from functools import partial

# Import the EchoTorch-specific components
from .non_local_coordinator import NonLocalCoordinator
from .experiment_echotorch import run_chsh_trial_echotorch
from .classical_system_echotorch import ClassicalSystemEchoTorch

# --- Configuration ---
ESN_DIMENSION = 100
HIDDEN_DIM = 256
MAX_GENERATIONS = 100
# Let's use a smaller population for faster iteration during performance tuning
POPULATION_SIZE = mp.cpu_count() if mp.cpu_count() <= 12 else 12

# --- Globals for Multiprocessing ---
# These will be initialized once per worker process
system = None
controller = None

def init_worker(device):
    """Initializes the simulation environment for each worker process."""
    global system, controller
    # Create a single, persistent ClassicalSystem instance for this worker
    system = ClassicalSystemEchoTorch(N=ESN_DIMENSION, device=device)
    # Create a single controller instance to be updated with weights
    controller = NonLocalCoordinator(
        esn_dimension=ESN_DIMENSION,
        hidden_dim=HIDDEN_DIM, 
        is_linear=False
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

def main(delay):
    """
    Main function using an efficient multiprocessing strategy.
    """
    print("--- Starting Project Apsu (EchoTorch): Phase 2 (Optimized) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We only need to get the number of parameters on the main process
    # The actual controller instances will be created in the workers
    temp_controller = NonLocalCoordinator(
        esn_dimension=ESN_DIMENSION, hidden_dim=HIDDEN_DIM, is_linear=False
    )
    n_params = sum(p.numel() for p in temp_controller.parameters())
    del temp_controller
    print(f"Optimizing a NON-LINEAR controller with {n_params} parameters.")

    cma_options = {
        'popsize': POPULATION_SIZE,
        'CMA_diagonal': True,
        'seed': 42
    }
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.5, cma_options)
    
    print(f"Running CMA-ES for {MAX_GENERATIONS} generations with population size {POPULATION_SIZE}.")

    best_s_scores = []
    
    # Use the 'spawn' context for CUDA safety
    # Initialize each worker process with the init_worker function
    with mp.get_context("spawn").Pool(initializer=init_worker, initargs=(device,)) as pool:
        with tqdm(range(MAX_GENERATIONS), desc="CMA-ES Generations") as pbar:
            for gen in pbar:
                solutions = es.ask()
                
                # Use partial to pre-fill the non-changing arguments
                partial_eval = partial(evaluate_fitness, device=device, delay=delay)
                
                # The pool will now efficiently reuse the workers and their environments
                fitnesses = pool.map(partial_eval, solutions)
                
                es.tell(solutions, fitnesses)
                
                best_s_in_gen = -es.result.fbest
                best_s_scores.append(best_s_in_gen)
                
                pbar.set_postfix({"best_S": f"{best_s_in_gen:.4f}"})

    print("\nOptimization complete.")
    print(f"Best S-score found: {-es.result.fbest:.4f}")

    # Plotting and saving results...
    plt.figure(figsize=(10, 6))
    plt.plot(best_s_scores, marker='o')
    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.axhline(2.828, color='g', linestyle='--', label="Tsirelson's Bound (S≈2.828)")
    plt.title("Best S-Score vs. Generation (Non-Linear Controller, EchoTorch, Optimized)")
    plt.xlabel("Generation")
    plt.ylabel("Best S-Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = "apsu/phase2_nonlinear_controller_results_echotorch.png"
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")
    plt.close()
    
    if max(best_s_scores) > 2.02:
        print("\nSuccess Gate Passed: NON-LINEAR controller VIOLATED the classical bound!")
    else:
        print("\nSuccess Gate FAILED: Non-linear controller was unable to violate the classical bound.")
        
    print("--- Phase 2 (EchoTorch, Optimized) Complete ---")

if __name__ == "__main__":
    delay_to_test = 1
    print(f"--- Running experiment with controller delay d={delay_to_test} ---")
    main(delay=delay_to_test) 