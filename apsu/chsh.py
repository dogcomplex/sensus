import torch
import numpy as np
import logging
from .classical_system_echotorch import ClassicalSystemEchoTorch
from .non_local_coordinator import NonLocalCoordinator
from .utils import get_chsh_targets  # Import from the new utils file
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _create_controller(controller_config, system):
    """Factory function to create the NonLocalCoordinator."""
    # This function now acts as a strict gatekeeper, only passing expected
    # arguments to the constructor to prevent TypeErrors from legacy config keys.
    config = controller_config.get('config', {})
    n_inputs = system.n_units * 2
    
    # Explicitly build the config for the NonLocalCoordinator
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
    Computes the CHSH S-score from binarized outcomes.
    S = |E(a,b) - E(a,b')| + |E(a',b) + E(a',b')|
    """
    # This function assumes inputs are already binarized numpy arrays (+1/-1)
    outcomes_A = outputs_A
    outcomes_B = outputs_B

    correlations = {}
    
    # Calculate E(a, b) for all four setting combinations
    for sa_val, sb_val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mask = (a_settings == sa_val) & (b_settings == sb_val)
        if np.any(mask):
            correlation = np.mean(outcomes_A[mask] * outcomes_B[mask])
        else:
            correlation = 0.0
        
        # Map (0,0) -> 'ab', (0,1) -> 'ab_prime', etc.
        key = f"a{'_prime' if sa_val == 1 else ''}_b{'_prime' if sb_val == 1 else ''}"
        correlations[key] = correlation

    E_ab = correlations.get("a_b", 0)
    E_ab_prime = correlations.get("a_b_prime", 0)
    E_a_prime_b = correlations.get("a_prime_b", 0)
    E_a_prime_b_prime = correlations.get("a_prime_b_prime", 0)
    
    S = abs(E_ab - E_ab_prime) + abs(E_a_prime_b + E_a_prime_b_prime)
    
    # For compatibility with the rest of the system's logging.
    final_correlations = {
        "C(a,b)": E_ab, "C(a,b')": E_ab_prime,
        "C(a',b)": E_a_prime_b, "C(a',b')": E_a_prime_b_prime
    }

    return S, final_correlations

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
    Evaluates the fitness of an individual using a scientifically robust
    two-system "air gap" protocol to prevent any possible state leakage.
    """
    try:
        # --- 1. Setup ---
        device = eval_config.get('device', 'cpu')
        system_config = eval_config.get('classical_system', {})
        controller_config = eval_config.get('controller', {})
        chsh_config = eval_config.get('chsh_evaluation', {})

        # Create two identical, separate systems. One for calibration, one for evaluation.
        system_cal = ClassicalSystemEchoTorch(device=device, **system_config)
        
        # The evaluation system MUST have a different seed to be scientifically valid.
        system_eval_config = system_config.copy()
        system_eval_config['seed'] = system_config.get('seed', 0) + 2
        system_eval = ClassicalSystemEchoTorch(device=device, **system_eval_config)

        controller = None
        if individual is not None:
            # The controller only ever interacts with the evaluation system.
            controller = _create_controller(controller_config, system_eval)
            controller.set_weights(individual)
            controller.to(device)
            controller.eval()

        washout_time = chsh_config.get('washout_time', 1000)
        eval_time = chsh_config.get('eval_time', 1000)
        chsh_seed = chsh_config.get('chsh_seed')
        if chsh_seed is None:
            chsh_seed = np.random.randint(0, 2**31 - 1)
            logging.warning(f"No 'chsh_seed' found. Using a random one: {chsh_seed}")

        # --- 2. Readout Calibration on the Calibration System ---
        total_cal_steps = washout_time + eval_time
        
        cal_states_A, cal_states_B = system_cal.run_and_collect_states(total_cal_steps)

        # CRITICAL FIX: The calibration seed MUST be cryptographically independent
        # of the evaluation seed. We generate a new, unpredictable seed for every
        # single fitness evaluation to break the super-deterministic link.
        # FIX: Explicitly use int64 to avoid overflow on 32-bit systems.
        cal_seed = np.random.randint(0, 2**31 - 1, dtype=np.int64)

        cal_settings_A, cal_settings_B = _get_chsh_settings(total_cal_steps, seed=cal_seed)
        cal_targets_A, cal_targets_B = get_chsh_targets(cal_settings_A, cal_settings_B, seed=cal_seed)

        system_cal.calibrate_readouts(
            cal_states_A[washout_time:], cal_states_B[washout_time:],
            cal_settings_A[washout_time:], cal_settings_B[washout_time:],
            cal_targets_A[washout_time:], cal_targets_B[washout_time:]
        )
        
        # --- 3. Transfer Weights to the Evaluation System ---
        trained_weights = system_cal.get_readout_weights()
        system_eval.set_readout_weights(trained_weights)

        # --- 4. Controller Evaluation on the Pristine Evaluation System ---
        if controller:
            # The NonLocalCoordinator is stateless, so no reset is needed.
            # This line is removed to prevent the AttributeError.
            pass
        
        eval_settings_A, eval_settings_B = _get_chsh_settings(eval_time, seed=chsh_seed)
        live_outputs_A, live_outputs_B = [], []
        
        with torch.no_grad():
            # Washout loop
            for _ in range(washout_time):
                state_A, state_B = system_eval.get_state_individual()
                full_state = torch.cat((state_A.flatten(), state_B.flatten())).unsqueeze(0)
                correction = controller(full_state) if controller else torch.zeros((1, 2), device=device)
                if correction.dim() > 1:
                    correction = correction.squeeze(0)
                system_eval.step(correction[0], correction[1])

            # Evaluation loop
            for k in range(eval_time):
                # First, get the current state for the controller to act upon.
                current_state_A, current_state_B = system_eval.get_state_individual()
                full_state = torch.cat((current_state_A.flatten(), current_state_B.flatten())).unsqueeze(0)
                
                # The controller computes the correction based on the current state.
                correction = controller(full_state) if controller else torch.zeros((1, 2), device=device)
                if correction.dim() > 1:
                    correction = correction.squeeze(0)
                
                # Now, step the system forward. The new state is what we will measure.
                new_state_A, new_state_B = system_eval.step(correction[0], correction[1])
                
                # Measure the outcome from the NEW state, after the controller's action.
                y_A, y_B = system_eval.get_live_outputs(new_state_A, new_state_B, eval_settings_A[k], eval_settings_B[k])
                live_outputs_A.append(y_A.item())
                live_outputs_B.append(y_B.item())

        # --- 5. Scoring ---
        outcomes_A = np.sign(live_outputs_A)
        outcomes_B = np.sign(live_outputs_B)
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