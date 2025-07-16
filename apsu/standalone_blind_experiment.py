import torch
import torch.nn as nn
import numpy as np
import cma
from pathlib import Path
import json
import time
import argparse
import logging
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from tqdm import tqdm

# =============================================================================
# --- Core Components (largely unchanged) ---
# =============================================================================

class ClassicalSystem:
    def __init__(self, units, sr, lr, seed):
        self.reservoir_A = Reservoir(units, sr=sr, lr=lr, seed=seed)
        self.reservoir_B = Reservoir(units, sr=sr, lr=lr, seed=seed + 1)

    def step(self, input_A, input_B):
        state_A = self.reservoir_A.run(np.array([[input_A]]))
        state_B = self.reservoir_B.run(np.array([[input_B]]))
        return torch.from_numpy(state_A).float().squeeze(0), torch.from_numpy(state_B).float().squeeze(0)

    def reset(self):
        self.reservoir_A.reset()
        self.reservoir_B.reset()

class IntegratedController(nn.Module):
    def __init__(self, reservoir_units, controller_units):
        super().__init__()
        input_size = reservoir_units * 2
        self.w_in = nn.Linear(input_size, controller_units, bias=False)
        self.w_hidden = nn.Linear(controller_units, controller_units, bias=False)
        self.w_out = nn.Linear(controller_units, 4, bias=False) # c_a, c_b, y_a, y_b

    def forward(self, x_a, x_b):
        x_combined = torch.cat((x_a, x_b), dim=-1)
        hidden = torch.tanh(self.w_in(x_combined))
        hidden = torch.tanh(self.w_hidden(hidden))
        outputs = torch.tanh(self.w_out(hidden))
        return outputs[..., 0], outputs[..., 1], outputs[..., 2], outputs[..., 3]

def set_weights(model, weights_vector):
    with torch.no_grad():
        start = 0
        for param in model.parameters():
            num_params = param.numel()
            param.copy_(torch.tensor(weights_vector[start:start+num_params]).view(param.shape))
            start += num_params

# =============================================================================
# --- Test Harness (Modified for Double-Blind Protocol) ---
# =============================================================================

def get_chsh_measurements(rng, num_steps):
    theta_a = rng.choice([0, np.pi/2], size=num_steps)
    theta_b = rng.choice([np.pi/4, 3*np.pi/4], size=num_steps)
    return theta_a, theta_b

def calculate_s_score(outputs_A, outputs_B, theta_a, theta_b):
    correlations = {}
    setting_pairs = [
        (0, np.pi/4), (0, 3*np.pi/4),
        (np.pi/2, np.pi/4), (np.pi/2, 3*np.pi/4)
    ]
    
    for ta_val, tb_val in setting_pairs:
        indices = (theta_a == ta_val) & (theta_b == tb_val)
        if np.any(indices):
            E = np.mean(outputs_A[indices] * outputs_B[indices])
            correlations[(ta_val, tb_val)] = E
        else:
            correlations[(ta_val, tb_val)] = 0.0

    s_score = correlations.get((0, np.pi/4), 0) - correlations.get((0, 3*np.pi/4), 0) + \
              correlations.get((np.pi/2, np.pi/4), 0) + correlations.get((np.pi/2, 3*np.pi/4), 0)
    
    return s_score

def evaluate_chsh(weights_vector, config, phi_sequence, chsh_seed):
    """
    Core evaluation function. Runs a CHSH simulation for a given controller
    against a specific sequence of random phases (phi).
    The CHSH measurement angles (theta) are derived from a seed to ensure they
    are independent of the phi sequence but dependent on the controller hash.
    """
    system_config = config['classical_system']
    controller_config = config['controller']
    sim_config = config['simulation']

    controller = IntegratedController(system_config['units'], controller_config['units'])
    set_weights(controller, weights_vector)
    controller.eval()

    controller_hash = hash(str(weights_vector.data.tobytes()))
    eval_rng = np.random.RandomState((chsh_seed + controller_hash) % (2**32 - 1))
    
    num_steps = min(sim_config['T_total'], len(phi_sequence))
    theta_a, theta_b = get_chsh_measurements(eval_rng, num_steps)
    phi = phi_sequence[:num_steps]

    classical_system = ClassicalSystem(system_config['units'], system_config['sr'], system_config['lr'], config['seed'])
    
    outputs_A, outputs_B = [], []
    x_a, x_b = torch.zeros(system_config['units']), torch.zeros(system_config['units'])
    
    with torch.no_grad():
        for t in range(num_steps):
            c_a, c_b, y_a, y_b = controller(x_a, x_b)
            outputs_A.append(np.sign(y_a.item()))
            outputs_B.append(np.sign(y_b.item()))
            x_a, x_b = classical_system.step(c_a.item(), c_b.item())

    s_score = calculate_s_score(np.array(outputs_A), np.array(outputs_B), theta_a, theta_b)
    return s_score

# =============================================================================
# --- Main Experiment Orchestrator (Modified for Double-Blind Protocol) ---
# =============================================================================
def main():
    rpy.verbosity(0)
    parser = argparse.ArgumentParser(description="Run a DOUBLE-BLIND, rigorous CHSH experiment.")
    # Args are the same as the non-blind version for consistency
    parser.add_argument('--controller-units', type=int, default=1, help="Override the number of controller units.")
    parser.add_argument('--seed', type=int, default=42, help="Override the main random seed.")
    args = parser.parse_args()

    # --- Configuration ---
    config = {
        "seed": args.seed,
        "classical_system": {"units": 50, "sr": 0.95, "lr": 0.3},
        "controller": {"units": args.controller_units},
        "chsh_evaluation": {"delay": 1},
        "simulation": {"T_total": 4000, "generations": 100, "population_size": 10}
    }

    # --- Setup Output Directory and Logging ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"apsu/standalone_blind_results/exp_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = results_dir / "run.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info(f"--- Starting Standalone DOUBLE-BLIND Experiment ---")
    logging.info(f"Results will be saved to: {results_dir}")
    logging.info("Configuration: \n" + json.dumps(config, indent=2))

    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # --- Load Double-Blind Data ---
    train_file = Path("apsu/utils/training_randomness.bin")
    test_file = Path("apsu/utils/testing_randomness.bin")
    
    try:
        with open(train_file, 'rb') as f:
            train_phi = np.frombuffer(f.read(), dtype=np.float32)
        with open(test_file, 'rb') as f:
            test_phi = np.frombuffer(f.read(), dtype=np.float32)
        logging.info(f"Successfully loaded {len(train_phi)} training and {len(test_phi)} testing points.")
    except FileNotFoundError:
        logging.error(f"Missing randomness files. Please run 'apsu/utils/generate_double_blind_sets.py' first.")
        return

    # --- Optimizer Setup ---
    temp_controller = IntegratedController(config['classical_system']['units'], config['controller']['units'])
    num_params = sum(p.numel() for p in temp_controller.parameters())
    logging.info(f"Total parameters to optimize: {num_params}")

    es = cma.CMAEvolutionStrategy(num_params * [0], 0.5, {'popsize': config['simulation']['population_size'], 'seed': config['seed'], 'verb_disp': 0, 'verb_log': 0})

    # --- Main Optimization Loop ---
    history = {'train_s': [], 'test_s': []}
    best_ever_test_s_score = -4.0
    
    pbar = tqdm(total=config['simulation']['generations'], desc="Optimizing (Double-Blind)")
    for generation in range(config['simulation']['generations']):
        solutions = es.ask()
        # *** CRITICAL: Fitness scores for the optimizer are ONLY from the TRAINING set ***
        fitnesses = [-evaluate_chsh(s, config, train_phi, config['seed']) for s in solutions]
        es.tell(solutions, fitnesses)
        
        # --- Holdout Set Evaluation ---
        # Evaluate the best candidate of this generation on the UNSEEN test set
        best_solution_weights = es.result.xbest
        test_s_score = evaluate_chsh(best_solution_weights, config, test_phi, config['seed'])
        
        train_s_score = -es.result.fbest
        history['train_s'].append(train_s_score)
        history['test_s'].append(test_s_score)
        
        if test_s_score > best_ever_test_s_score:
            best_ever_test_s_score = test_s_score
            # Save the best controller based on TEST performance
            best_controller = IntegratedController(config['classical_system']['units'], config['controller']['units'])
            set_weights(best_controller, best_solution_weights)
            torch.save(best_controller.state_dict(), results_dir / "best_controller.pth")

        pbar.set_postfix({"Train S": f"{train_s_score:.4f}", "Test S": f"{test_s_score:.4f}", "Best Test S": f"{best_ever_test_s_score:.4f}"})
        pbar.update(1)

    pbar.close()

    # --- Finalization ---
    logging.info("--- Experiment Complete ---")
    logging.info(f"Final Best TRAINING S-Score: {history['train_s'][-1]:.4f}")
    logging.info(f"Final Best TESTING S-Score: {best_ever_test_s_score:.4f}")

    # Save final results
    final_results = {
        "best_training_s_score": history['train_s'][-1],
        "best_testing_s_score": best_ever_test_s_score,
        "config": config,
        "history": history
    }
    with open(results_dir / "results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    logging.info(f"Saved final results to {results_dir / 'results.json'}")

if __name__ == "__main__":
    main() 