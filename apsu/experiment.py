import numpy as np
import torch

from .classical_system import ClassicalSystem
from . import chsh

# Parameters for the experiment as per spec §4.3
T_WASHOUT = 1000  # Steps to let the reservoir settle
T_EVAL = 4000     # Steps to evaluate for each CHSH setting combination

def run_chsh_trial(controller, seed, config=None):
    """
    Executes a single, complete CHSH experiment trial for a given controller.

    Args:
        controller (torch.nn.Module or None): The controller to test. If None,
                                              it runs a null experiment.
        seed (int): Random seed for this specific run.
        config (dict, optional): A dictionary for future configuration.

    Returns:
        float: The final calculated S-score for this trial.
    """
    # 1. Setup Phase
    system = ClassicalSystem(seed=seed)
    
    # Generate the stream of settings for Alice and Bob for this run
    alice_settings, bob_settings = chsh.get_chsh_settings(T_EVAL, seed=seed)
    
    # The base inputs to the reservoirs are simply the CHSH settings {-1, 1}
    base_inputs_A = (2 * alice_settings - 1).reshape(-1, 1)
    base_inputs_B = (2 * bob_settings - 1).reshape(-1, 1)
    
    # 2. Simulation Phase
    # First, a washout period with zero input to let the reservoir settle
    washout_input = np.zeros((T_WASHOUT, 1))
    # The state after washout is the initial state for the evaluation run
    state_A, state_B = system.step(washout_input, washout_input)

    # Then, the evaluation period where states are collected
    for i in range(T_EVAL):
        c_a, c_b = np.array([[0.0]]), np.array([[0.0]]) # Default zero correction
        if controller is not None:
            # The controller gets a global view of the system state
            # Convert numpy states to torch tensors
            state_a_torch = torch.from_numpy(state_A).float()
            state_b_torch = torch.from_numpy(state_B).float()
            
            # Compute corrective signals
            c_a_torch, c_b_torch = controller(state_a_torch, state_b_torch)
            c_a = c_a_torch.detach().numpy()
            c_b = c_b_torch.detach().numpy()

        # The input at each step is the CHSH setting plus the controller's signal
        input_A_t = np.array([[base_inputs_A[i, 0]]]) + c_a
        input_B_t = np.array([[base_inputs_B[i, 0]]]) + c_b
        
        # Evolve the system and get the new state
        new_state_A, new_state_B = system.step(input_A_t, input_B_t)
        
        # Collect the *new* state for the readout training
        system.collect_state(new_state_A, new_state_B)
        
        # The new state becomes the current state for the next iteration
        state_A, state_B = new_state_A, new_state_B

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