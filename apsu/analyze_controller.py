import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import reservoirpy as rpy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Components (copied from standalone_experiment.py for self-containment) ---

class IntegratedController(nn.Module):
    def __init__(self, reservoir_units, controller_units):
        super().__init__()
        input_size = reservoir_units * 2  # State from both reservoirs
        self.w_in = nn.Linear(input_size, controller_units)
        self.w_hidden = nn.Linear(controller_units, controller_units)
        # 4 outputs: correction_A, correction_B, readout_A, readout_B
        self.w_out = nn.Linear(controller_units, 4)

    def forward(self, x_a, x_b):
        x_combined = torch.cat((x_a, x_b), dim=-1)
        hidden = torch.tanh(self.w_in(x_combined))
        hidden = torch.tanh(self.w_hidden(hidden))
        outputs = torch.tanh(self.w_out(hidden))
        return outputs[..., 0], outputs[..., 1], outputs[..., 2], outputs[..., 3]

class ClassicalSystem:
    def __init__(self, units, sr, lr, seed):
        self.reservoir_A = rpy.nodes.Reservoir(units=units, sr=sr, lr=lr, seed=seed)
        self.reservoir_B = rpy.nodes.Reservoir(units=units, sr=sr, lr=lr, seed=seed + 1)

    def step(self, input_A, input_B):
        state_A = self.reservoir_A.run(input_A)
        state_B = self.reservoir_B.run(input_B)
        return torch.from_numpy(state_A).float(), torch.from_numpy(state_B).float()

    def reset(self):
        self.reservoir_A.reset()
        self.reservoir_B.reset()

def chsh_target_logic(theta_a, theta_b, phi):
    cos_a = np.cos(theta_a - phi)
    cos_b = np.cos(theta_b - phi)
    target_A = 1.0 if cos_a > 0 else -1.0
    target_B = -1.0 if cos_b > 0 else 1.0
    return target_A, target_B

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
            correlations[(ta_val, tb_val)] = 0

    s_score = correlations[(0, np.pi/4)] - correlations[(0, 3*np.pi/4)] + \
              correlations[(np.pi/2, np.pi/4)] + correlations[(np.pi/2, 3*np.pi/4)]
    
    return s_score, correlations

def evaluate_chsh(controller, config, seed, quantum_randomness_path):
    T_total = config['simulation']['T_total']
    
    controller_hash = hash(str(controller.state_dict()))
    eval_rng = np.random.RandomState((seed + controller_hash) % (2**32 - 1))
    
    theta_a, theta_b = get_chsh_measurements(eval_rng, T_total)
    
    try:
        with open(quantum_randomness_path, 'rb') as f:
            raw_bytes = f.read(T_total * 4) # 4 bytes per float32
            if len(raw_bytes) < T_total * 4:
                raise ValueError("Not enough quantum randomness data available.")
            phi = np.frombuffer(raw_bytes, dtype=np.float32) % (2 * np.pi)
    except FileNotFoundError:
        logging.error(f"Quantum randomness file not found at {quantum_randomness_path}")
        phi = eval_rng.uniform(0, 2 * np.pi, T_total)

    classical_system = ClassicalSystem(
        units=config['classical_system']['units'],
        sr=config['classical_system']['sr'],
        lr=config['classical_system']['lr'],
        seed=seed
    )

    outputs_A, outputs_B = [], []
    trajectory = {
        'hidden': [], 'c_a': [], 'c_b': [], 'y_a': [], 'y_b': [],
        'target_a': [], 'target_b': []
    }

    x_a, x_b = torch.zeros(config['classical_system']['units']), torch.zeros(config['classical_system']['units'])
    for t in range(T_total):
        c_a, c_b, y_a, y_b = controller(x_a, x_b)
        
        outputs_A.append(y_a.item())
        outputs_B.append(y_b.item())

        target_a, target_b = chsh_target_logic(theta_a[t], theta_b[t], phi[t])

        # Record trajectory
        with torch.no_grad():
             x_combined = torch.cat((x_a, x_b), dim=-1)
             hidden_state = torch.tanh(controller.w_in(x_combined))
        trajectory['hidden'].append(hidden_state.numpy().flatten())
        trajectory['c_a'].append(c_a.item())
        trajectory['c_b'].append(c_b.item())
        trajectory['y_a'].append(y_a.item())
        trajectory['y_b'].append(y_b.item())
        trajectory['target_a'].append(target_a)
        trajectory['target_b'].append(target_b)

        x_a, x_b = classical_system.step(c_a.detach(), c_b.detach())

    s_score, _ = calculate_s_score(np.array(outputs_A), np.array(outputs_B), theta_a, theta_b)
    return s_score, trajectory

def find_best_run(controller_units):
    results_dir = Path("apsu/standalone_results")
    best_run = None
    max_s_score = -5.0

    if not results_dir.exists():
        logging.error(f"Results directory '{results_dir}' not found.")
        return None

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        try:
            with open(exp_dir / "results.json") as f:
                results = json.load(f)
            
            if results['config']['controller']['units'] == controller_units:
                if results['best_s_score'] > max_s_score:
                    max_s_score = results['best_s_score']
                    best_run = exp_dir
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue

    if best_run:
        logging.info(f"Found best run for {controller_units} units in '{best_run}' with S-Score: {max_s_score:.4f}")
    else:
        logging.warning(f"No valid runs found for {controller_units} units.")
        
    return best_run


def plot_trajectory(trajectory, run_dir):
    logging.info("Plotting controller trajectory...")
    num_steps = len(trajectory['c_a'])
    t = np.arange(num_steps)

    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Controller Trajectory Analysis - {run_dir.name}", fontsize=16)

    # Panel 1: Controller's internal state
    hidden_states = np.array(trajectory['hidden'])
    for i in range(hidden_states.shape[1]):
        axs[0].plot(t, hidden_states[:, i], label=f'Hidden Unit {i+1}', alpha=0.8)
    axs[0].set_title("Controller Internal Hidden State(s)")
    axs[0].set_ylabel("Activation")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Panel 2: Controller's corrective outputs
    axs[1].plot(t, trajectory['c_a'], label='Correction A', color='r', alpha=0.8)
    axs[1].plot(t, trajectory['c_b'], label='Correction B', color='b', alpha=0.8)
    axs[1].set_title("Controller Corrective Signals")
    axs[1].set_ylabel("Signal Value")
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    
    # Panel 3: Predicted vs Target Outputs
    axs[2].plot(t, trajectory['y_a'], label='Predicted Y_A', color='orange', linestyle='--', alpha=0.7)
    axs[2].plot(t, trajectory['target_a'], label='Target Y_A', color='orange', alpha=0.4)
    axs[2].plot(t, trajectory['y_b'], label='Predicted Y_B', color='purple', linestyle='--', alpha=0.7)
    axs[2].plot(t, trajectory['target_b'], label='Target Y_B', color='purple', alpha=0.4)
    axs[2].set_title("Controller Predictions vs. Ground Truth Targets")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Measurement Output")
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = run_dir / "trajectory_analysis.png"
    plt.savefig(plot_path)
    logging.info(f"Saved trajectory plot to {plot_path}")
    plt.close()


def plot_weights(controller, run_dir):
    logging.info("Plotting controller weights...")
    w_in = controller.w_in.weight.detach().numpy().T
    w_out = controller.w_out.weight.detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Controller Weight Analysis - {run_dir.name}", fontsize=16)

    # Input weights
    axs[0].set_title("Input Weights (Reservoir State -> Hidden Unit)")
    cax = axs[0].imshow(w_in, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=axs[0], label="Weight Value")
    axs[0].set_xlabel("Hidden Unit")
    axs[0].set_ylabel("Reservoir Neuron Index (0-49: A, 50-99: B)")
    axs[0].axhline(y=49.5, color='r', linestyle='--')


    # Output weights
    axs[1].set_title("Output Weights (Hidden Unit -> Outputs)")
    cax = axs[1].imshow(w_out, cmap='viridis', aspect='auto')
    fig.colorbar(cax, ax=axs[1], label="Weight Value")
    axs[1].set_yticks(np.arange(4))
    axs[1].set_yticklabels(['Correction A', 'Correction B', 'Readout A', 'Readout B'])
    axs[1].set_xlabel("Hidden Unit")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = run_dir / "weight_analysis.png"
    plt.savefig(plot_path)
    logging.info(f"Saved weight plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze a trained controller to reverse-engineer its function.")
    parser.add_argument('--controller-units', type=int, default=1, help="Number of controller units for the run to analyze.")
    parser.add_argument('--run-dir', type=str, help="Specify a run directory to analyze directly.")
    args = parser.parse_args()

    rpy.verbosity(0)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_best_run(args.controller_units)

    if not run_dir or not run_dir.exists():
        logging.error(f"Could not find a valid run directory. Exiting.")
        return

    # --- Load everything from the specified run ---
    logging.info(f"--- Analyzing run: {run_dir.name} ---")
    
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    
    controller_path = run_dir / "best_controller.pth"
    if not controller_path.exists():
        logging.error(f"Controller weights not found at {controller_path}")
        return

    controller = IntegratedController(
        reservoir_units=config['classical_system']['units'],
        controller_units=config['controller']['units']
    )
    controller.load_state_dict(torch.load(controller_path))
    controller.eval()

    quantum_file = Path("apsu/utils/ANU_quantum_randomness.bin")

    # --- Analysis 1: The "Frozen Questions" Test (The Smoking Gun) ---
    logging.info("--- Running 'Frozen Questions' Test ---")
    original_seed = config['seed']
    mismatched_seed = original_seed + 1337

    s_score_original, trajectory = evaluate_chsh(controller, config, original_seed, quantum_file)
    logging.info(f"S-Score with ORIGINAL seed ({original_seed}): {s_score_original:.4f} (Expected: ~{results['best_s_score']:.4f})")

    s_score_mismatched, _ = evaluate_chsh(controller, config, mismatched_seed, quantum_file)
    logging.info(f"S-Score with MISMATCHED seed ({mismatched_seed}): {s_score_mismatched:.4f} (Hypothesis: ~2.0)")

    if s_score_original > 2.8 and s_score_mismatched < 2.1:
        logging.info("SUCCESS: Test confirms hypothesis. High score is seed-specific.")
    else:
        logging.warning("WARNING: Test results are inconclusive. The controller might be more general than hypothesized.")

    # --- Analysis 2: Trajectory Visualization ---
    plot_trajectory(trajectory, run_dir)
    
    # --- Analysis 3: Weight Visualization ---
    plot_weights(controller, run_dir)

    logging.info("--- Analysis Complete ---")

if __name__ == "__main__":
    main() 