import torch
import numpy as np
import logging
from .classical_system_echotorch import ClassicalSystemEchoTorch
from .non_local_coordinator import NonLocalCoordinator
from .reservoir_controller import ReservoirController
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _create_controller(config, device):
    """Factory function to create the appropriate controller."""
    controller_config = config.get('controller', {})
    controller_type = controller_config.get('type', 'NonLinear')

    # Correctly infer input dimension from the system configuration
    system_config = config.get('classical_system', config.get('classical_system_config', {}))
    system_units = system_config.get('units', 100) # Default to 100 if not found
    default_input_dim = 2 * system_units

    if controller_type == 'NonLinear':
        input_dim = controller_config.get('input_dim', default_input_dim)
        return NonLocalCoordinator(
            input_dim=input_dim,
            hidden_dim=controller_config.get('hidden_dim', 16),
            output_dim=2,
            use_bias=controller_config.get('use_bias', True)
        ).to(device)
    elif controller_type == 'Reservoir':
        input_dim = controller_config.get('input_dim', default_input_dim)
        # The reservoir controller's config is nested one level deeper
        rc_config = controller_config.get('config', {})
        return ReservoirController(
            input_dim=input_dim,
            output_dim=2,
            **rc_config
        ).to(device)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

def _compute_s_score(outputs_A, outputs_B, settings_A, settings_B):
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

    for i in range(len(settings_A)):
        product = outcomes_A_bin[i] * outcomes_B_bin[i]
        if settings_A[i] == 0 and settings_B[i] == 0:
            outcomes["ab"].append(product)
        elif settings_A[i] == 0 and settings_B[i] == 1:
            outcomes["ab_prime"].append(product)
        elif settings_A[i] == 1 and settings_B[i] == 0:
            outcomes["a_prime_b"].append(product)
        elif settings_A[i] == 1 and settings_B[i] == 1:
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


def evaluate_fitness(weights, config, chsh_seed, return_diagnostics=False):
    """
    Evaluates the fitness of a given set of controller weights.
    """
    try:
        # --- 1. Setup ---
        # This section is now robust to different config structures from different runners.
        device = config.get('device', 'cpu')

        system_config = config.get('classical_system', config.get('classical_system_config', {}))
        controller_config = config.get('controller', {})
        eval_config = config.get('chsh_evaluation', config.get('simulation_config', {}))

        system = ClassicalSystemEchoTorch(device=device, **system_config)
        
        # Pass the full controller config to the factory
        controller = None
        if weights is not None:
            # Pass the full config object to the factory
            controller = _create_controller(config, device)
            controller.set_weights(weights)
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

        chsh_settings = _generate_chsh_data(n_bits=total_time, seed=chsh_seed)
        
        input_A_stream = torch.tensor(chsh_settings['settings_A'], dtype=torch.float32).view(-1, 1)
        input_B_stream = torch.tensor(chsh_settings['settings_B'], dtype=torch.float32).view(-1, 1)
        
        correction_buffer = torch.zeros((delay, 2), device=device)

        system.reset()

        with torch.no_grad():
            for k in range(total_time):
                state_A, state_B = system.get_state_individual()
                
                correction = torch.zeros((1, 2), device=device)
                if controller is not None:
                    full_state = torch.cat((state_A.flatten(), state_B.flatten())).unsqueeze(0)
                    correction = controller(full_state)
                
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
        readout_targets_A = chsh_settings['y_A'][washout_time:]
        readout_targets_B = chsh_settings['y_B'][washout_time:]
        readout_mse_A, readout_mse_B = system.train_readouts(readout_targets_A, readout_targets_B)

        # --- 4. Scoring ---
        y_pred_A, y_pred_B = system.get_readout_outputs()
        
        eval_settings_A = chsh_settings['settings_A'][washout_time:]
        eval_settings_B = chsh_settings['settings_B'][washout_time:]
        
        s_score, correlations = _compute_s_score(y_pred_A, y_pred_B, eval_settings_A, eval_settings_B)

        if return_diagnostics:
            return {
                "fitness": s_score,
                "s_value": s_score,
                "correlations": correlations,
                "readout_mse": (readout_mse_A + readout_mse_B) / 2
            }
        
        return s_score

    except Exception as e:
        logging.error(f"Error in fitness evaluation for seed {chsh_seed}: {e}", exc_info=True)
        # Return a very poor fitness score
        if return_diagnostics:
            return {"fitness": -1.0, "s_value": -1.0}
        return -1.0