import numpy as np
import torch
from torch.amp import autocast

from .classical_system_echotorch import ClassicalSystemEchoTorch
from . import chsh

# Parameters for the experiment as per spec §4.3
T_WASHOUT = 1000
T_EVAL = 4000

def run_chsh_trial_echotorch(controller, system, seed, device, delay=1):
    """
    Executes a single, complete, and VECTORIZED CHSH experiment trial
    using a pre-existing ClassicalSystem instance.
    """
    # 1. Setup Phase is now minimal, as the system is pre-initialized.
    #    We just need to reset the states and readouts for a clean run.
    system.readout_A.reset()
    system.readout_B.reset()
    system.states_A = []
    system.states_B = []
    
    # Generate the stream of settings for Alice and Bob for this run
    alice_settings, bob_settings = chsh.get_chsh_settings(T_EVAL, seed=seed)
    
    # Base inputs are the CHSH settings {-1, 1} as a tensor sequence
    # Shape: [Time, Features] -> [4000, 1]
    base_inputs_A = torch.tensor((2 * alice_settings - 1), dtype=torch.float32, device=device).reshape(-1, 1)
    base_inputs_B = torch.tensor((2 * bob_settings - 1), dtype=torch.float32, device=device).reshape(-1, 1)
    
    # 2. Vectorized Simulation Phase
    
    # A. Pre-computation pass to get the uncorrected state evolution.
    # We create a washout sequence and concatenate it with the real inputs.
    washout_inputs = torch.zeros(1, T_WASHOUT, 1, device=device, dtype=torch.float32)
    
    # Reshape inputs for ESNCell: (batch, time, features)
    pre_run_inputs_A = torch.cat([washout_inputs.squeeze(0), base_inputs_A]).unsqueeze(0)
    pre_run_inputs_B = torch.cat([washout_inputs.squeeze(0), base_inputs_B]).unsqueeze(0)
    
    # Run the reservoirs to get the full state sequence x(k)
    system.reservoir_A.reset_hidden()
    system.reservoir_B.reset_hidden()
    full_states_A = system.reservoir_A(pre_run_inputs_A).squeeze(0) # Shape [5000, 100]
    full_states_B = system.reservoir_B(pre_run_inputs_B).squeeze(0) # Shape [5000, 100]
    
    # We only need the states from the evaluation period for the controller
    uncorrected_states_A = full_states_A[T_WASHOUT:]
    uncorrected_states_B = full_states_B[T_WASHOUT:]
    
    # B. Compute all corrective signals in one go.
    with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
        corrections_A, corrections_B = controller(uncorrected_states_A, uncorrected_states_B)

    # C. Apply the delay (latency shim) to the corrective signals using tensor ops.
    if delay > 0:
        zero_padding = torch.zeros(delay, 1, device=device, dtype=corrections_A.dtype)
        delayed_corrections_A = torch.cat([zero_padding, corrections_A[:-delay]], dim=0)
        delayed_corrections_B = torch.cat([zero_padding, corrections_B[:-delay]], dim=0)
    else:
        delayed_corrections_A = corrections_A
        delayed_corrections_B = corrections_B
            
    # D. Create the final, corrected input sequences.
    final_inputs_A = base_inputs_A + delayed_corrections_A
    final_inputs_B = base_inputs_B + delayed_corrections_B

    # E. Run the definitive, fully-corrected simulation.
    # We must collect the states from this run for the readout.
    system.reservoir_A.reset_hidden()
    system.reservoir_B.reset_hidden()
    _ = system.reservoir_A(washout_inputs)
    _ = system.reservoir_B(washout_inputs)
    
    # Now run the evaluation part and collect the states
    eval_states_A = system.reservoir_A(final_inputs_A.unsqueeze(0)).squeeze(0)
    eval_states_B = system.reservoir_B(final_inputs_B.unsqueeze(0)).squeeze(0)

    # 3. Readout Training Phase - FOR FOUR INDEPENDENT READOUTS
    targets_by_setting = chsh.get_chsh_targets_by_setting(alice_settings, bob_settings)
    system.train_four_readouts(eval_states_A, eval_states_B, targets_by_setting)

    # 4. Scoring Phase - EVALUATE EACH SETTING INDEPENDENTLY
    correlations = {}
    setting_map = {'00': 0, '01': 1, '10': 2, '11': 3} # Map from string key to index

    for s_a in [0, 1]:
        for s_b in [0, 1]:
            setting_key = f"{s_a}{s_b}"
            setting_index = setting_map[setting_key]
            
            # Get the subset of data corresponding to this setting
            mask = (alice_settings == s_a) & (bob_settings == s_b)
            
            # Get outputs from the specific readout trained for this setting
            # We must pass the *full* state history to the readout, which will then give predictions.
            # Then we select the predictions corresponding to the mask.
            full_outputs_A, full_outputs_B = system.get_readout_outputs_for_setting(eval_states_A, eval_states_B, setting_index)

            outputs_A_setting = full_outputs_A[mask]
            outputs_B_setting = full_outputs_B[mask]
            
            # Ensure we have data to calculate correlation
            if len(outputs_A_setting) > 0:
                # Correlation is the expectation value of the product of outcomes
                correlation = np.mean(np.sign(outputs_A_setting) * np.sign(outputs_B_setting))
                correlations[setting_key] = correlation
            else:
                correlations[setting_key] = 0.0 # No data for this setting combination

    # Calculate the S-score from the four individually-calculated correlations
    s_score = correlations.get('00', 0) + correlations.get('01', 0) + correlations.get('10', 0) - correlations.get('11', 0)

    # Clean up collected states to free memory
    system.states_A.clear()
    system.states_B.clear()
    
    return s_score 