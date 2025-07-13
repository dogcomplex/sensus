import numpy as np
import torch
from collections import deque

from .classical_system_gpu import ClassicalSystemGPU
from . import chsh

# Parameters for the experiment as per spec §4.3
T_WASHOUT = 1000  # Steps to let the reservoir settle
T_EVAL = 4000     # Steps to evaluate for each CHSH setting combination

def run_chsh_trial(controller, seed, device, delay=1, config=None):
    """
    Executes a single, complete CHSH experiment trial for a given controller.
    """
    # 1. Setup Phase
    system = ClassicalSystemGPU(seed=seed, device=device)
    
    # Generate the stream of settings for Alice and Bob for this run
    alice_settings, bob_settings = chsh.get_chsh_settings(T_EVAL, seed=seed)
    
    # Base inputs are the CHSH settings {-1, 1} as a tensor sequence
    base_inputs_A = torch.tensor((2 * alice_settings - 1), dtype=torch.float32, device=device).reshape(-1, 1)
    base_inputs_B = torch.tensor((2 * bob_settings - 1), dtype=torch.float32, device=device).reshape(-1, 1)
    
    # 2. Simulation Phase (Now fully vectorized)
    
    # A. First, run a "pre-computation" pass to get the uncorrected state evolution.
    #    This gives us the state sequence needed to calculate the controller's corrections.
    #    This pass includes the washout period.
    washout_inputs_A = torch.zeros(T_WASHOUT, 1, device=device)
    washout_inputs_B = torch.zeros(T_WASHOUT, 1, device=device)
    
    pre_run_inputs_A = torch.cat([washout_inputs_A, base_inputs_A])
    pre_run_inputs_B = torch.cat([washout_inputs_B, base_inputs_B])
    
    # Run the reservoirs with washout + real inputs to get the state sequence x(k)
    # The controller will need the states from the evaluation period only.
    full_states_A, _ = system.reservoir_A.run(pre_run_inputs_A)
    full_states_B, _ = system.reservoir_B.run(pre_run_inputs_B)
    
    uncorrected_states_A = full_states_A[T_WASHOUT:]
    uncorrected_states_B = full_states_B[T_WASHOUT:]
    
    # B. Compute all corrective signals in one go.
    if controller is not None:
        # The controller gets the entire history of states
        corrections_A, corrections_B = controller(uncorrected_states_A, uncorrected_states_B)
        
        # C. Apply the delay (latency shim) to the corrective signals.
        # We shift the corrections forward in time and pad the start with zeros.
        if delay > 0:
            # Create zero-padding for the start of the sequence
            zero_padding_A = torch.zeros(delay, corrections_A.shape[1], device=device)
            zero_padding_B = torch.zeros(delay, corrections_B.shape[1], device=device)
            
            # Prepend padding and truncate the end to maintain sequence length
            delayed_corrections_A = torch.cat([zero_padding_A, corrections_A[:-delay]], dim=0)
            delayed_corrections_B = torch.cat([zero_padding_B, corrections_B[:-delay]], dim=0)
        else:
            delayed_corrections_A = corrections_A
            delayed_corrections_B = corrections_B
            
        # D. Create the final, corrected input sequences.
        final_inputs_A = base_inputs_A + delayed_corrections_A
        final_inputs_B = base_inputs_B + delayed_corrections_B
    else:
        # If no controller, the final inputs are just the base inputs
        final_inputs_A = base_inputs_A
        final_inputs_B = base_inputs_B

    # E. Run the definitive, fully-corrected simulation.
    system.run_and_collect(final_inputs_A, final_inputs_B, washout_steps=T_WASHOUT)

    # 3. Readout Training Phase
    # Generate the 'ideal' QM targets for the given settings
    targets_A, targets_B = chsh.get_chsh_targets(alice_settings, bob_settings)
    
    # Train the readouts to best fit the collected states to the targets
    system.train_readouts(targets_A, targets_B)
    
    # 4. Scoring Phase
    # Get the actual outputs from the trained readouts
    outputs_A, outputs_B = system.get_readout_outputs()

    # Calculate the S-score based on the system's actual behavior
    s_score = chsh.calculate_s_score(outputs_A, outputs_B, alice_settings, bob_settings)
    
    return s_score 