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
from reservoirpy.nodes import Reservoir  # Using real reservoir components

# =============================================================================
# --- 1. The Physical System (The "Slow Medium") ---
# This section defines the classical substrate using real reservoirpy components.
# =============================================================================

class ClassicalSystem:
    def __init__(self, units, sr, lr, seed):
        # Use a real Reservoir node from reservoirpy
        self.reservoir_A = Reservoir(units, sr=sr, lr=lr, seed=seed)
        self.reservoir_B = Reservoir(units, sr=sr, lr=lr, seed=seed + 1)

    def step(self, input_A, input_B):
        # The state is returned as a numpy array by reservoirpy
        state_A = self.reservoir_A.run(input_A)
        state_B = self.reservoir_B.run(input_B)
        return state_A, state_B

    def reset(self):
        self.reservoir_A.reset()
        self.reservoir_B.reset()

# =============================================================================
# --- 2. The Controller & Readout (The "Fast Medium" / The Agent) ---
# This is the entity we are optimizing. It now includes its own readout layers.
# The optimizer must learn both the control policy AND how to read the state.
# =============================================================================

class IntegratedController(nn.Module):
    def __init__(self, reservoir_units, controller_units):
        super().__init__()
        input_dim = reservoir_units * 2
        
        # Non-linear controller policy
        self.policy = nn.Sequential(
            nn.Linear(input_dim, controller_units),
            nn.ReLU(),
            nn.Linear(controller_units, controller_units),
            nn.ReLU(),
            nn.Linear(controller_units, 2), # Corrective signals c_A, c_B
            nn.Tanh()
        )
        
        # A single, universal linear readout for all settings.
        self.readout = nn.Linear(input_dim, 2)
        
    def forward(self, state_A, state_B):
        combined_state = torch.cat([state_A, state_B], dim=1)
        
        # Get corrective signal
        corrections = self.policy(combined_state)
        
        # Get measurement output from the single, universal readout
        measurement = self.readout(combined_state)
        
        # Return a tuple containing four references to the *same* measurement.
        # This enforces that the agent cannot know which setting is being applied.
        return corrections, (measurement, measurement, measurement, measurement)

# Helper to manage weights for the nn.Module
def set_weights(model, weights_vector):
    with torch.no_grad():
        start = 0
        for param in model.parameters():
            num_params = param.numel()
            param.copy_(torch.tensor(weights_vector[start:start+num_params]).view(param.shape))
            start += num_params

def get_param_count(model):
    return sum(p.numel() for p in model.parameters())

# =============================================================================
# --- 3. The Test Harness (The Objective Function) ---
# This is the "black box" environment. It takes a controller, runs the CHSH
# simulation, and returns a single fitness score.
# =============================================================================

def get_chsh_targets(settings, N):
    """Generates the 'correct' answers for a given setting."""
    targets_A = np.zeros(N)
    targets_B = np.zeros(N)
    for i in range(N):
        alpha, beta = settings[i]
        targets_A[i] = 1 if alpha == 0 else -1
        targets_B[i] = 1 if beta == 0 else -1
    return targets_A, targets_B

def evaluate_fitness(weights_vector, config, random_settings):
    # 1. Setup
    system_config = config['classical_system']
    controller_config = config['controller']

    system = ClassicalSystem(
        units=system_config['units'],
        sr=system_config['sr'],
        lr=system_config['lr'],
        seed=config['seed']
    )
    
    controller = IntegratedController(
        reservoir_units=system_config['units'],
        controller_units=controller_config['units']
    )
    set_weights(controller, weights_vector)
    controller.eval()
    
    # 2. Simulation
    T_total = config['simulation']['T_total']
    delay = config['chsh_evaluation']['delay']
    correction_buffer = torch.zeros((delay, 2))
    
    outputs_per_setting = ([], [], [], [])
    
    system.reset()
    with torch.no_grad():
        for t in range(T_total):
            # Get CHSH setting, using cyclic access
            alpha, beta = random_settings[t % len(random_settings)]
            setting_idx = alpha * 2 + beta

            # Inputs are based on CHSH settings
            input_A = np.array([[1.0 if alpha == 0 else -1.0]])
            input_B = np.array([[1.0 if beta == 0 else -1.0]])

            # Apply delayed correction from buffer
            delayed_correction = correction_buffer[0].numpy()
            
            # Evolve system
            state_A_np, state_B_np = system.step(
                input_A + delayed_correction[0],
                input_B + delayed_correction[1]
            )
            
            state_A = torch.from_numpy(state_A_np).float()
            state_B = torch.from_numpy(state_B_np).float()
            
            # Get new corrections and measurements from controller
            corrections, all_measurements = controller(state_A, state_B)
            
            # Update correction buffer
            correction_buffer = torch.roll(correction_buffer, shifts=-1, dims=0)
            correction_buffer[-1] = corrections
            
            # Store the measurement corresponding to the current setting
            outputs_per_setting[setting_idx].append(all_measurements[setting_idx].numpy())

    # 3. Scoring
    correlations = []
    for i in range(4):
        if not outputs_per_setting[i]: continue
        y = np.array([o[0] for o in outputs_per_setting[i]])
        targets_A, targets_B = get_chsh_targets(random_settings[0:len(y)], len(y))
        y_A = np.sign(y[:, 0])
        y_B = np.sign(y[:, 1])
        correlation = np.mean(y_A * y_B)
        correlations.append(correlation)

    if len(correlations) == 4:
        S = correlations[0] - correlations[1] + correlations[2] + correlations[3]
    else:
        S = -4.0 # Invalid run

    return -S


# =============================================================================
# --- 4. The Main Experiment Orchestrator ---
# =============================================================================
def main():
    # Silence reservoirpy to prevent verbose outputs during runs
    rpy.verbosity(0)

    parser = argparse.ArgumentParser(description="Run a standalone, rigorous CHSH experiment.")
    parser.add_argument('--config', type=str, help="Path to a JSON config file.")
    args = parser.parse_args()

    # --- Configuration ---
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config if none is provided
        config = {
            "seed": 42,
            "classical_system": {"units": 50, "sr": 0.95, "lr": 0.3},
            "controller": {"units": 32},
            "chsh_evaluation": {"delay": 1},
            "simulation": {"T_total": 4000, "generations": 100, "population_size": 10}
        }
    
    # --- Setup Output Directory and Logging ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"apsu/standalone_results/exp_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / "run.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info(f"--- Starting Standalone Experiment ---")
    logging.info(f"Results will be saved to: {output_dir}")
    logging.info("Configuration: \n" + json.dumps(config, indent=2))

    # Save config to output directory
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Load quantum randomness
    random_file = Path("apsu/experiments/qrng_chsh_settings.bin")
    if not random_file.exists():
        logging.error(f"Missing randomness file: {random_file}. Please run fetch_randomness.py.")
        raise FileNotFoundError(f"Missing randomness file: {random_file}")
    raw_random_data = np.fromfile(random_file, dtype=np.int32).reshape(-1, 2)
    random_settings = raw_random_data % 2
    
    # Initialize controller to get parameter count
    temp_controller = IntegratedController(
        config['classical_system']['units'], 
        config['controller']['units']
    )
    num_params = get_param_count(temp_controller)
    logging.info(f"Total parameters to optimize: {num_params}")

    # --- Optimization ---
    es = cma.CMAEvolutionStrategy(num_params * [0], 0.5, {'popsize': config['simulation']['population_size'], 'seed': config['seed']})
    
    start_time = time.time()
    s_score_history = []
    for gen in range(config['simulation']['generations']):
        solutions = es.ask()
        fitnesses = [evaluate_fitness(s, config, random_settings) for s in solutions]
        es.tell(solutions, fitnesses)
        
        best_s_score = -es.result.fbest
        s_score_history.append(best_s_score)
        logging.info(f"Generation {gen+1}/{config['simulation']['generations']} | Best S-Score: {best_s_score:.4f}")

    # --- Results ---
    end_time = time.time()
    logging.info("\n--- Optimization Finished ---")
    logging.info(f"Total time: {end_time - start_time:.2f} seconds")
    logging.info(f"Best ever S-Score found: {-es.result.fbest:.4f}")
    
    # Save best weights
    best_weights = es.result.xbest
    np.save(output_dir / "best_controller_weights.npy", best_weights)
    logging.info(f"Saved best weights to {output_dir / 'best_controller_weights.npy'}")

    # Save final results to a structured JSON file for easy analysis
    final_results = {
        "best_s_score": float(-es.result.fbest),
        "s_score_history": [float(s) for s in s_score_history],
        "controller_units": config['controller']['units'],
        "total_generations": es.result.iterations,
        "total_time_seconds": end_time - start_time,
        "config": config
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    logging.info(f"Saved final results to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main() 