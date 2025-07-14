import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging

from .utils import get_chsh_targets
from reservoirpy.nodes import Reservoir, Ridge

class ClassicalSystemReservoirPy:
    """
    Manages the two ESN reservoirs and their corresponding readout layers.
    This version has been refactored to use the reservoirpy library to
    ensure a robust, independent implementation.
    """
    def __init__(self, units, spectral_radius, leaking_rate, input_scaling, noise_rc, seed, device):
        # reservoirpy is numpy-based, so the device parameter is not used.
        self.device = 'cpu' 
        self.n_units = units
        self.seed = seed

        # Create reservoirs using reservoirpy
        self.reservoir_A = Reservoir(units, lr=leaking_rate, sr=spectral_radius,
                                     noise_rc=noise_rc, input_scaling=input_scaling, seed=seed)
        self.reservoir_B = Reservoir(units, lr=leaking_rate, sr=spectral_radius,
                                     noise_rc=noise_rc, input_scaling=input_scaling, seed=seed+1)

        # Create FOUR independent readout layers using reservoirpy's Ridge node
        # A small ridge parameter is added for numerical stability.
        self.readouts_A = [Ridge(ridge=1e-7) for _ in range(4)]
        self.readouts_B = [Ridge(ridge=1e-7) for _ in range(4)]

    def reset(self):
        # reservoirpy nodes are stateful and are reset implicitly when .run() is called
        # on a new sequence. No explicit reset is needed for this design.
        pass

    def step(self, input_A, input_B):
        # For reservoirpy, we typically run over sequences, not single steps.
        # This method is adapted to be stateful for the evaluation loop.
        state_A = self.reservoir_A.run(np.array([[input_A]]))
        state_B = self.reservoir_B.run(np.array([[input_B]]))
        return state_A, state_B

    def get_state_individual(self):
        # Return the last internal state
        return self.reservoir_A.state(), self.reservoir_B.state()

    def run_and_collect_states(self, total_steps, noise_level=0.1):
        # Create a noise generator with a specific seed for reproducibility
        rng = np.random.default_rng(self.seed)
        noise_inputs = rng.standard_normal((total_steps, 1)) * noise_level

        # Run the reservoirs over the full noise sequence
        states_A = self.reservoir_A.run(noise_inputs, reset=True)
        states_B = self.reservoir_B.run(noise_inputs, reset=True)
        return states_A, states_B
        
    def calibrate_readouts(self, cal_states_A, cal_states_B, cal_settings_A, cal_settings_B, cal_targets_A, cal_targets_B):
        # Reshape targets for reservoirpy
        cal_targets_A = cal_targets_A.reshape(-1, 1)
        cal_targets_B = cal_targets_B.reshape(-1, 1)

        for i in range(4):
            s_a, s_b = i // 2, i % 2
            mask = (cal_settings_A == s_a) & (cal_settings_B == s_b)
            if not np.any(mask): continue
            
            # Use the simple .fit() method from reservoirpy
            self.readouts_A[i].fit(cal_states_A[mask], cal_targets_A[mask])
            self.readouts_B[i].fit(cal_states_B[mask], cal_targets_B[mask])

    def get_live_outputs(self, state_A, state_B, setting_A, setting_B):
        setting_index = setting_A * 2 + setting_B
        
        readout_A = self.readouts_A[setting_index]
        readout_B = self.readouts_B[setting_index]
        
        # Use the .run() method to get the output
        output_A = readout_A.run(state_A)
        output_B = readout_B.run(state_B)
        
        return output_A.flatten(), output_B.flatten()

    def diagnose(self, *args, **kwargs):
        # Placeholder to avoid breaking the harness
        pass 