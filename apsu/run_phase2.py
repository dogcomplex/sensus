import cma
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import reservoirpy as rpy
import torch.multiprocessing as mp
from functools import partial

from .non_local_coordinator import NonLocalCoordinator
from .experiment import run_chsh_trial

# Silence ReservoirPy's verbose output
rpy.verbosity(0)
rpy.set_seed(42)

# --- Configuration ---
ESN_DIMENSION = 100
HIDDEN_DIM = 256
MAX_GENERATIONS = 100 # Let's run for longer, this is the main event
POPULATION_SIZE = 24  # A larger population for a more complex search space

def set_weights(model, weights):
    """Injects a flat vector of weights into a PyTorch model."""
    with torch.no_grad():
        start = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data.copy_(torch.tensor(weights[start:start+num_params]).view_as(param))
            start += num_params

def evaluate_fitness(weights, device, delay):
    """
    Fitness function for the CMA-ES optimizer.
    It takes a weight vector, runs a CHSH trial, and returns the S-score.
    CMA-ES minimizes, so we return -S.
    """
    controller = NonLocalCoordinator(
        esn_dimension=ESN_DIMENSION, 
        hidden_dim=HIDDEN_DIM, 
        is_linear=False
    )
    set_weights(controller, weights)
    controller.to(device) # Move the controller to the selected device
    
    # OPTIMIZATION: Convert model to half-precision for Tensor Core acceleration
    if device.type == 'cuda':
        controller.half()

    # We run one trial per fitness evaluation for speed.
    # The seed is kept constant to have a static fitness landscape.
    s_score = run_chsh_trial(controller, seed=42, device=device, delay=delay)
    
    # CMA-ES minimizes, but we want to maximize S.
    return -s_score

def main(delay):
    """
    Main function to execute Phase 2: Linear Controller Validation.
    """
    print("--- Starting Project Apsu: Phase 2 (Linear Controller Test) ---")

    # 1. Setup device for training (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("NOTE: CUDA not available. Running on CPU will be significantly slower.")


    # 2. Initialize the model to find the number of parameters
    controller = NonLocalCoordinator(
        esn_dimension=ESN_DIMENSION, 
        hidden_dim=HIDDEN_DIM, 
        is_linear=False
    )
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Optimizing a NON-LINEAR controller with {n_params} parameters.")

    # 3. Set up the CMA-ES Optimizer
    # Start with zero weights and a standard deviation of 0.5
    # Per spec §7.1 and memory constraints, use a diagonal covariance matrix.
    # This is crucial for high-dimensional search spaces.
    cma_options = {
        'popsize': POPULATION_SIZE,
        'CMA_diagonal': True,
        'seed': 42
    }
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.5, cma_options)
    
    print(f"Running CMA-ES for {MAX_GENERATIONS} generations with population size {POPULATION_SIZE}.")
    print("Using DIAGONAL covariance matrix to conserve memory.")

    best_s_scores = []
    
    # 4. Run the Optimization Loop
    with tqdm(range(MAX_GENERATIONS), desc="CMA-ES Generations") as pbar:
        for gen in pbar:
            solutions = es.ask()
            
            # Parallelize fitness evaluations to saturate the GPU
            with mp.get_context("spawn").Pool() as pool:
                # 'partial' pre-fills the 'device' argument for the fitness function
                partial_eval = partial(evaluate_fitness, device=device, delay=delay)
                fitnesses = pool.map(partial_eval, solutions)
            
            es.tell(solutions, fitnesses)
            
            best_s_in_gen = -es.result.fbest
            best_s_scores.append(best_s_in_gen)
            
            pbar.set_postfix({"best_S": f"{best_s_in_gen:.4f}"})

    print("\nOptimization complete.")
    print(f"Best S-score found: {es.result.fbest:.4f}")

    # 4. Save results and plot
    plt.figure(figsize=(10, 6))
    plt.plot(best_s_scores, marker='o')
    plt.axhline(2.0, color='r', linestyle='--', label="Classical Bound (S=2)")
    plt.axhline(2.828, color='g', linestyle='--', label="Tsirelson's Bound (S≈2.828)")
    plt.title("Best S-Score vs. Generation (Non-Linear Controller)")
    plt.xlabel("Generation")
    plt.ylabel("Best S-Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = "apsu/phase2_nonlinear_controller_results.png"
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")
    plt.close()
    
    # 5. Check Success Gate
    if max(best_s_scores) > 2.02:
        print("\nSuccess Gate Passed: NON-LINEAR controller VIOLATED the classical bound!")
    else:
        print("\nSuccess Gate FAILED: Non-linear controller was unable to violate the classical bound.")
        
    print("--- Phase 2 (Non-Linear Test) Complete ---")

if __name__ == "__main__":
    # This is crucial for multiprocessing with CUDA on some platforms
    mp.freeze_support() 
    
    # --- Experiment Configuration ---
    # Per spec §2.3, we must test different delay values.
    # We start with d=1 as the baseline.
    delay_to_test = 1
    print(f"--- Running experiment with controller delay d={delay_to_test} ---")

    main(delay=delay_to_test) 