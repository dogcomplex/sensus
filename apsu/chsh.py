import torch
import numpy as np
import logging
from .classical_system_echotorch import ClassicalSystemEchoTorch
from .non_local_coordinator import NonLocalCoordinator
from .reservoir_controller import ReservoirController
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _create_controller(controller_config, system):
    """Factory function to create a controller based on config."""
    controller_type = controller_config.get('type', 'NonLocal') # Default to NonLocal
    config = controller_config.get('config', {})

    if controller_type.lower() == 'reservoir':
        # Ensure reservoir_config is passed if nested
        if 'reservoir_config' in config:
            config = config['reservoir_config']
        return ReservoirController(**config)
    
    elif controller_type.lower() == 'nonlocal':
        # Pass the system's state dimension to the NLC
        n_inputs = system.units * 2 
        return NonLocalCoordinator(n_inputs=n_inputs, **config)
    
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def _get_chsh_settings(n_steps, seed=None, settings_path=None):
    """
    Generates or loads the random settings for the CHSH test.
    Returns two numpy arrays (a_settings, b_settings) of 0s and 1s.
    """
    if settings_path and os.path.exists(settings_path):
        # Load from a pre-generated file for maximum rigor
        # The file is expected to be a simple text file with two columns of 0s and 1s
        settings = np.loadtxt(settings_path, dtype=int)
        if len(settings) < n_steps:
             raise ValueError(f"Random settings file {settings_path} is too short.")
        return settings[:n_steps, 0], settings[:n_steps, 1]
    else:
        # Generate pseudo-randomly for standard runs
        rng = np.random.default_rng(seed)
        a_settings = rng.integers(0, 2, size=n_steps)
        b_settings = rng.integers(0, 2, size=n_steps)
        return a_settings, b_settings

def _compute_s_score(outputs_A, outputs_B, a_settings, b_settings):
    """
    Computes the CHSH S-score.
    S = |E(a,b) - E(a,b')| + |E(a',b) + E(a',b')|
    """
    # Ensure outputs are numpy arrays on the CPU for calculation
    if isinstance(outputs_A, torch.Tensor):
        outputs_A = outputs_A.squeeze().cpu().numpy()
    if isinstance(outputs_B, torch.Tensor):
        outputs_B = outputs_B.squeeze().cpu().numpy()
    
    # Binarize the outputs to +1/-1, which is required for a valid CHSH calculation.
    # The raw outputs of the linear readout are not bounded.
    outcomes_A_bin = np.sign(outputs_A)
    outcomes_B_bin = np.sign(outputs_B)

    # Ensure there are no zeros from the sign function, map them to +1
    outcomes_A_bin[outcomes_A_bin == 0] = 1
    outcomes_B_bin[outcomes_B_bin == 0] = 1

    outcomes = {
        "ab": [], "ab_prime": [], "a_prime_b": [], "a_prime_b_prime": []
    }

    for i in range(len(a_settings)):
        product = outcomes_A_bin[i] * outcomes_B_bin[i]
        if a_settings[i] == 0 and b_settings[i] == 0:
            outcomes["ab"].append(product)
        elif a_settings[i] == 0 and b_settings[i] == 1:
            outcomes["ab_prime"].append(product)
        elif a_settings[i] == 1 and b_settings[i] == 0:
            outcomes["a_prime_b"].append(product)
        elif a_settings[i] == 1 and b_settings[i] == 1:
            outcomes["a_prime_b_prime"].append(product)

    E_ab = np.mean(outcomes["ab"]) if outcomes["ab"] else 0
    E_ab_prime = np.mean(outcomes["ab_prime"]) if outcomes["ab_prime"] else 0
    E_a_prime_b = np.mean(outcomes["a_prime_b"]) if outcomes["a_prime_b"] else 0
    E_a_prime_b_prime = np.mean(outcomes["a_prime_b_prime"]) if outcomes["a_prime_b_prime"] else 0
    
    correlations = {
        "C(a,b)": E_ab, "C(a,b')": E_ab_prime,
        "C(a',b)": E_a_prime_b, "C(a',b')": E_a_prime_b_prime
    }
    S = abs(E_ab - E_ab_prime) + abs(E_a_prime_b + E_a_prime_b_prime)
    return S, correlations

def _generate_chsh_data(n_bits, seed):
    """Generates the measurement settings and ideal target outputs for a CHSH trial."""
    np.random.seed(seed)
    settings_A = np.random.randint(0, 2, n_bits)
    settings_B = np.random.randint(0, 2, n_bits)
    
    y_A = np.zeros(n_bits)
    y_B = np.zeros(n_bits)
    
    # Standard QM predictions for CHSH with optimal angles
    for i in range(n_bits):
        if settings_A[i] == 0 and settings_B[i] == 0: # a, b
            y_A[i], y_B[i] = 1, np.cos(np.pi/4)
        elif settings_A[i] == 0 and settings_B[i] == 1: # a, b'
            y_A[i], y_B[i] = 1, np.cos(-np.pi/4)
        elif settings_A[i] == 1 and settings_B[i] == 0: # a', b
            y_A[i], y_B[i] = 0, np.cos(np.pi/4)
        elif settings_A[i] == 1 and settings_B[i] == 1: # a', b'
            y_A[i], y_B[i] = 0, np.cos(-np.pi/4)
            
    return {"settings_A": settings_A, "settings_B": settings_B, "y_A": y_A, "y_B": y_B}


def evaluate_fitness(individual, eval_config, return_full_results=False):
    """
    Evaluates the fitness of an individual (a set of controller weights).
    """
    try:
        # --- 1. Setup ---
        # This section is now robust to different config structures from different runners.
        device = eval_config.get('device', 'cpu')

        system_config = eval_config.get('classical_system', eval_config.get('classical_system_config', {}))
        controller_config = eval_config.get('controller', {})
        eval_config = eval_config.get('chsh_evaluation', eval_config.get('simulation_config', {}))

        system = ClassicalSystemEchoTorch(device=device, **system_config)
        
        # Pass the full controller config to the factory
        controller = None
        if individual is not None:
            # Pass the full config object to the factory
            controller = _create_controller(controller_config, system)
            controller.set_weights(individual)
            controller.to(device)
            controller.eval()

        # --- 2. Simulation ---
        delay_float = eval_config.get('delay', eval_config.get('controller_delay', 0))
        # Ensure delay is an integer for buffer creation.
        # This will treat d=0.5 as d=0 for now, avoiding the crash.
        delay = int(delay_float)
        washout_time = eval_config.get('washout_time', 1000)
        eval_time = eval_config.get('eval_time', eval_config.get('eval_block_size', 1000))
        total_time = washout_time + eval_time

        # --- 3. Get CHSH Settings ---
        chsh_seed = eval_config.get('chsh_seed', None)
        settings_path = eval_config.get('chsh_settings_path', None)
        a_settings, b_settings = _get_chsh_settings(total_time, seed=chsh_seed, settings_path=settings_path)
        
        # --- 4. Simulation Loop ---
        system.reset()
        controller.reset()

        input_A_stream = torch.tensor(a_settings, dtype=torch.float32).view(-1, 1)
        input_B_stream = torch.tensor(b_settings, dtype=torch.float32).view(-1, 1)
        
        correction_buffer = torch.zeros((delay, 2), device=device)

        with torch.no_grad():
            for k in range(total_time):
                state_A, state_B = system.get_state_individual()
                
                correction = torch.zeros((1, 2), device=device)
                if controller is not None:
                    full_state = torch.cat((state_A.flatten(), state_B.flatten())).unsqueeze(0)
                    correction = controller(full_state)
                
                # --- HOSTILE ABLATION CHECK ---
                # If ablation is active, ignore the controller's output and use zeros.
                if eval_config.get('ablate_controller', False):
                    correction = torch.zeros_like(correction)
                # --- END ABLATION CHECK ---
                
                if delay > 0:
                    delayed_correction = correction_buffer[0, :]
                    correction_buffer = torch.roll(correction_buffer, shifts=-1, dims=0)
                    correction_buffer[-1, :] = correction
                else:
                    # If there's no delay, the "delayed" correction is the current one.
                    delayed_correction = correction.squeeze()
                
                input_A = input_A_stream[k].unsqueeze(0) + delayed_correction[0].item()
                input_B = input_B_stream[k].unsqueeze(0) + delayed_correction[1].item()
                
                system.step(input_A, input_B, collect=(k >= washout_time))

        # --- 3. Readout Training ---
        readout_targets_A = a_settings[washout_time:]
        readout_targets_B = b_settings[washout_time:]
        readout_mse_A, readout_mse_B = system.train_readouts(readout_targets_A, readout_targets_B)

        # --- 4. Scoring ---
        y_pred_A, y_pred_B = system.get_readout_outputs()
        
        eval_settings_A = a_settings[washout_time:]
        eval_settings_B = b_settings[washout_time:]
        
        s_score, correlations = _compute_s_score(y_pred_A, y_pred_B, eval_settings_A, eval_settings_B)
        
        if return_full_results:
            return {
                "fitness": s_score if s_score is not None else -1.0,
                "s_value": s_score,
                "correlations": correlations
            }
        return s_score if s_score is not None else -1.0

    except Exception as e:
        # Use a generic error message since chsh_seed may not be available
        logging.error(f"Error in fitness evaluation: {e}", exc_info=True)
        # Return a very poor fitness score
        if return_full_results:
            return {"fitness": -1.0, "s_value": -1.0, "correlations": {}}
        return -1.0