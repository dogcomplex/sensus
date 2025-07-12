import cma
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import reservoirpy as rpy

from .non_local_coordinator import NonLocalCoordinator
from .experiment import run_chsh_trial

# Silence ReservoirPy's verbose output
rpy.verbosity(0)
rpy.set_seed(42)

# --- Configuration ---
ESN_DIMENSION = 100
MAX_GENERATIONS = 50
POPULATION_SIZE = 12 

def set_weights(model, weights):
    """Injects a flat vector of weights into a PyTorch model."""
    with torch.no_grad():
        start = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data.copy_(torch.tensor(weights[start:start+num_params]).view_as(param))
            start += num_params

def evaluate_fitness(weights):
    """
    Fitness function for the CMA-ES optimizer.
    It takes a weight vector, runs a CHSH trial, and returns the S-score.
    CMA-ES minimizes, so we return -S.
    """
    controller = NonLocalCoordinator(esn_dimension=ESN_DIMENSION, is_linear=True)
    set_weights(controller, weights)
    
    # We run one trial per fitness evaluation for speed.
    # The seed is kept constant to have a static fitness landscape.
    s_score = run_chsh_trial(controller, seed=42)
    
    # CMA-ES minimizes, but we want to maximize S.
    return -s_score

def main():
    """
    Main function to execute Phase 2: Linear Controller Validation.
    """
    print("--- Starting Project Apsu: Phase 2 (Linear Controller Test) ---")

    # 1. Initialize the model to find the number of parameters
    controller = NonLocalCoordinator(esn_dimension=ESN_DIMENSION, is_linear=True)
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Optimizing a linear controller with {n_params} parameters.")

    # 2. Set up the CMA-ES Optimizer
    # Start with zero weights and a standard deviation of 0.5
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.5, {'popsize': POPULATION_SIZE})
    
    print(f"Running CMA-ES for {MAX_GENERATIONS} generations with population size {POPULATION_SIZE}.")

    best_s_scores = []
    
    # 3. Run the Optimization Loop
    with tqdm(range(MAX_GENERATIONS), desc="CMA-ES Generations") as pbar:
        for gen in pbar:
            solutions = es.ask()
            
            # For now, we run evaluations serially. Can be parallelized later.
            fitnesses = [evaluate_fitness(s) for s in solutions]
            
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
    plt.title("Best S-Score vs. Generation (Linear Controller)")
    plt.xlabel("Generation")
    plt.ylabel("Best S-Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = "apsu/phase2_linear_controller_results.png"
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")
    plt.close()
    
    # 5. Check Success Gate
    if max(best_s_scores) <= 2.02:
        print("\nSuccess Gate Passed: Linear controller was unable to violate the classical bound.")
    else:
        print("\nSuccess Gate FAILED: Linear controller appears to have violated the classical bound.")
        
    print("--- Phase 2 (Linear Test) Complete ---")

if __name__ == "__main__":
    main() 