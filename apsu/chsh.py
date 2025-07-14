import torch
import numpy as np
import logging
from .classical_system_reservoirpy import ClassicalSystemReservoirPy
from .non_local_coordinator import NonLocalCoordinator
from .utils.core_utils import get_chsh_targets
import matplotlib.pyplot as plt
import os
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _create_controller(controller_config, system):
    """Factory function to create the NonLocalCoordinator."""
    config = controller_config.get('config', {})
    n_inputs = system.n_units * 2 
    
    nlc_config = {
        'input_dim': n_inputs,
        'hidden_dim': config.get('hidden_dim'),
        'output_dim': config.get('output_dim', 2),
        'use_bias': config.get('use_bias', True)
    }

    if nlc_config['hidden_dim'] is None:
        raise ValueError("Controller config in the experiment file is missing the required 'hidden_dim' key.")

    return NonLocalCoordinator(**nlc_config)


def _get_chsh_settings(n_steps, seed=None, settings_path=None):
    """
    Get CHSH settings (a, b) for a given number of steps.

    This function now prioritizes loading from a pre-generated binary file
    of true quantum random numbers. If the file doesn't exist, it falls back
    to a pseudo-random generator but issues a strong warning.
    """
    if not settings_path or not os.path.exists(settings_path):
        raise FileNotFoundError(
            f"Quantum randomness file not found at '{settings_path}'. "
            "Please run the main batch_runner.py script to fetch the file "
            "from the ANU API. If the file exists, check the path."
        )

    with open(settings_path, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Each step needs 2 bits (1 for Alice, 1 for Bob).
    bits_needed = n_steps * 2
    bytes_needed = (bits_needed + 7) // 8 # Ceiling division
    
    if len(raw_bytes) < bytes_needed:
        raise ValueError(f"Randomness file {settings_path} is too small. "
                         f"Needs {bytes_needed} bytes, but only has {len(raw_bytes)}. "
                         "Re-run the fetcher script with a larger --bytes value.")

    bit_stream = np.unpackbits(raw_bytes)
    
    a_settings = bit_stream[:n_steps]
    b_settings = bit_stream[n_steps:n_steps*2]
    
    return a_settings, b_settings

def _compute_s_score(outputs_A, outputs_B, a_settings, b_settings):
    outcomes_A = outputs_A
    outcomes_B = outputs_B

    correlations = {}
    
    for sa_val, sb_val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mask = (a_settings == sa_val) & (b_settings == sb_val)
        if np.any(mask):
            correlation = np.mean(outcomes_A[mask] * outcomes_B[mask])
        else:
            correlation = 0.0
        
        key = f"a{'_prime' if sa_val == 1 else ''}_b{'_prime' if sb_val == 1 else ''}"
        correlations[key] = correlation

    E_ab = correlations.get("a_b", 0)
    E_ab_prime = correlations.get("a_b_prime", 0)
    E_a_prime_b = correlations.get("a_prime_b", 0)
    E_a_prime_b_prime = correlations.get("a_prime_b_prime", 0)
    
    S = abs(E_ab - E_ab_prime) + abs(E_a_prime_b + E_a_prime_b_prime)
    
    final_correlations = {
        "C(a,b)": E_ab, "C(a,b')": E_ab_prime,
        "C(a',b)": E_a_prime_b, "C(a',b')": E_a_prime_b_prime
    }

    return S, final_correlations

def evaluate_fitness(individual, eval_config, return_full_results=False):
    try:
        device = eval_config.get('device', 'cpu')
        system_config = eval_config.get('classical_system', {})
        controller_config = eval_config.get('controller', {})
        chsh_config = eval_config.get('chsh_evaluation', {})

        # The new system uses reservoirpy and is much simpler.
        # The two-system "air gap" is no longer needed as .fit() provides separation.
        system = ClassicalSystemReservoirPy(device=device, **system_config)

        controller = _create_controller(controller_config, system)
        controller.set_weights(individual)
        controller.to(device)
        controller.eval()

        washout_time = chsh_config.get('washout_time', 1000)
        eval_time = chsh_config.get('eval_time', 1000)
        chsh_seed = chsh_config.get('chsh_seed')
        if chsh_seed is None:
            chsh_seed = np.random.randint(0, 2**31 - 1)
            logging.warning(f"No 'chsh_seed' found. Using a random one: {chsh_seed}")

        # Define the path to the true randomness file.
        q_settings_path = "apsu/experiments/qrng_chsh_settings.bin"

        # --- Calibration Phase ---
        total_cal_steps = washout_time + eval_time
        cal_states_A, cal_states_B = system.run_and_collect_states(total_cal_steps)

        # The seed for calibration is now effectively the content of the random file.
        cal_settings_A, cal_settings_B = _get_chsh_settings(total_cal_steps, seed=chsh_seed, settings_path=q_settings_path)
        cal_targets_A, cal_targets_B = get_chsh_targets(cal_settings_A, cal_settings_B, seed=chsh_seed)

        system.calibrate_readouts(
            cal_states_A[washout_time:], cal_states_B[washout_time:],
            cal_settings_A[washout_time:], cal_settings_B[washout_time:],
            cal_targets_A[washout_time:], cal_targets_B[washout_time:]
        )
        
        # --- Evaluation Phase ---
        system.reset()
        eval_settings_A, eval_settings_B = _get_chsh_settings(eval_time, seed=chsh_seed, settings_path=q_settings_path)
        live_outputs_A, live_outputs_B = [], []
        
        # Washout loop
        with torch.no_grad():
            for _ in range(washout_time):
                state_A, state_B = system.get_state_individual()
                # Convert numpy states to torch tensors for the controller
                full_state = torch.from_numpy(np.concatenate([state_A.flatten(), state_B.flatten()])).float().unsqueeze(0).to(device)
                correction = controller(full_state)
                # Convert torch corrections back to numpy for the system
                correction_np = correction.squeeze(0).cpu().numpy()
                system.step(correction_np[0], correction_np[1])

            # Evaluation loop
            for k in range(eval_time):
                # 1. Get the state x(k)
                current_state_A, current_state_B = system.get_state_individual()
                
                # 2. Measure the output y(k) based on the CURRENT state x(k)
                y_A, y_B = system.get_live_outputs(current_state_A, current_state_B, eval_settings_A[k], eval_settings_B[k])
                live_outputs_A.append(y_A)
                live_outputs_B.append(y_B)
                
                # 3. Compute correction c(k) based on x(k)
                full_state = torch.from_numpy(np.concatenate([current_state_A.flatten(), current_state_B.flatten()])).float().unsqueeze(0).to(device)
                correction = controller(full_state)
                correction_np = correction.squeeze(0).cpu().numpy()

                # 4. Evolve the system to the next state x(k+1) using the correction
                system.step(correction_np[0], correction_np[1])

        # --- Scoring ---
        outcomes_A = np.sign(np.concatenate(live_outputs_A))
        outcomes_B = np.sign(np.concatenate(live_outputs_B))
        outcomes_A[outcomes_A == 0] = 1
        outcomes_B[outcomes_B == 0] = 1

        s_score, correlations = _compute_s_score(outcomes_A, outcomes_B, eval_settings_A, eval_settings_B)

        if return_full_results:
            return {
                "fitness": s_score if s_score is not None else -1.0,
                "s_value": s_score,
                "correlations": correlations
            }
        return s_score if s_score is not None else -1.0

    except Exception as e:
        logging.error(f"Error in fitness evaluation: {e}", exc_info=True)
        if return_full_results:
            return {"fitness": -2.0, "s_value": -2.0, "correlations": {}}
        return -2.0

def calculate_s_score(outputs_A, outputs_B, alice_settings, bob_settings):
    """
    Calculates the CHSH S-score from the system's outputs.
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

    for i in range(len(alice_settings)):
        product = outcomes_A_bin[i] * outcomes_B_bin[i]
        if alice_settings[i] == 0 and bob_settings[i] == 0:
            outcomes["ab"].append(product)
        elif alice_settings[i] == 0 and bob_settings[i] == 1:
            outcomes["ab_prime"].append(product)
        elif alice_settings[i] == 1 and bob_settings[i] == 0:
            outcomes["a_prime_b"].append(product)
        elif alice_settings[i] == 1 and bob_settings[i] == 1:
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