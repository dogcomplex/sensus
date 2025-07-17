import torch
import numpy as np
from collections import Counter
import logging

from apsu6.substrate import ClassicalSubstrate
from apsu6.controller import UniversalController
from apsu6.metrics import calculate_chsh_score, calculate_nonsignaling_metric
from apsu6.utils import load_chsh_settings, bits_to_spins

class DelayBuffer:
    """Handles the delay `d` for the controller's correction signals."""
    def __init__(self, delay: int, num_channels: int = 2, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        """If delay is 0, the buffer acts as a passthrough."""
        self.delay = int(delay)
        if self.delay < 0:
            raise ValueError("Delay cannot be negative.")
        self.num_channels = num_channels
        self.buffer = torch.zeros(self.delay, self.num_channels, device=device, dtype=dtype)
    
    def push(self, signal: torch.Tensor) -> torch.Tensor:
        """Pushes a new signal in, returns the oldest one."""
        assert signal.numel() == self.num_channels or signal.shape[0] == 1, "Signal shape must be compatible."
        signal_1d = signal.detach().squeeze().view(self.num_channels)
        if self.delay == 0:
            return signal_1d
        
        oldest_signal = self.buffer[0, :].clone()
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        self.buffer[-1, :] = signal_1d
        return oldest_signal
        
    def reset(self):
        self.buffer.zero_()

class ExperimentHarness:
    """
    The main orchestration script. Sets up the experiment, runs the simulation 
    loop, gathers data, and computes the final fitness score.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        self.substrate = ClassicalSubstrate(**config['substrate_params'], device=self.device)
        self.controller = UniversalController(**config['controller_params'], device=self.device)
        
        controller_delay = config.get('controller_delay', 1.0)
        self.delay_buffer = DelayBuffer(
            delay=int(controller_delay) if controller_delay >= 1 else 0, 
            device=self.device
        )
        
        self.chsh_settings = load_chsh_settings(config.get('randomness_file'))
        self.noise_rng = torch.Generator(device=self.device)
        self.noise_rng.manual_seed(config.get('noise_seed', 42))

    def _build_input(self, setting_bit: float, correction_val: float) -> np.ndarray:
        """Composes the substrate drive vector [setting_bit, correction_val]."""
        return np.array([[setting_bit, correction_val]], dtype=np.float32)

    def evaluate_fitness(self, controller_weights: dict, sensor_noise_std: float = 0.0) -> tuple[float, dict]:
        """
        Performs one full fitness evaluation for a given set of controller weights.
        The noise level can be set dynamically for curriculum learning.
        """
        self.substrate.reset()
        self.controller.reset()
        self.delay_buffer.reset()
        self.controller.load_state_dict(controller_weights)
        self.controller.eval()
        
        # sensor_noise_std is now passed in as an argument.
        actuation_scale = self.config.get('actuation_scale', 1.0)
        
        T_total = self.config.get('T_total', 4000)
        washout_steps = self.config.get('washout_steps', 100)
        
        # The number of scored trials is T_total from the config
        num_scored_trials = T_total
        # The settings file must contain exactly this many settings
        assert len(self.chsh_settings) == num_scored_trials, "CHSH settings file length must match T_total."

        outputs_A, outputs_B, settings_log = [], [], []

        with torch.no_grad():
            # --- 1. Pre-Washout Phase ---
            # Run the system for `washout_steps` to let transients settle.
            # Use placeholder (0,0) inputs; these are not scored.
            logging.debug(f"Starting washout phase for {washout_steps} steps...")
            placeholder_setting_tensor = torch.tensor([[1.0]], device=self.device, dtype=torch.float32) # Spin for +1
            for _ in range(washout_steps):
                state_A, state_B = self.substrate.get_state()
                
                # Controller runs, but its output is only used to drive the substrate
                correction_logits = self.controller.forward(state_A, state_B, placeholder_setting_tensor, placeholder_setting_tensor)
                delayed_logits = self.delay_buffer.push(correction_logits)
                delayed_logit_A, delayed_logit_B = delayed_logits[0], delayed_logits[1]

                # Evolve substrate
                substrate_input_A = self._build_input(0.0, delayed_logit_A.item() * actuation_scale)
                substrate_input_B = self._build_input(0.0, delayed_logit_B.item() * actuation_scale)
                self.substrate.step(substrate_input_A, substrate_input_B)
            logging.debug("Washout complete.")

            # --- 2. Scored Evaluation Phase ---
            # Now, iterate through the entire balanced CHSH settings file.
            logging.debug(f"Starting scored evaluation for {num_scored_trials} steps...")
            for t in range(num_scored_trials):
                # 2a. Get current state and apply sensor noise for controller
                state_A, state_B = self.substrate.get_state() # clean state, shape (B, N)
                
                noisy_state_A = state_A + torch.randn(state_A.shape, device=self.device, generator=self.noise_rng) * sensor_noise_std
                noisy_state_B = state_B + torch.randn(state_B.shape, device=self.device, generator=self.noise_rng) * sensor_noise_std
                
                # 2b. Get CHSH settings for this time step from the balanced file
                setting_A_bit, setting_B_bit = self.chsh_settings[t]
                setting_A_spin, setting_B_spin = bits_to_spins((setting_A_bit, setting_B_bit))
                
                setting_A_tensor = torch.tensor([[setting_A_spin]], device=self.device, dtype=torch.float32)
                setting_B_tensor = torch.tensor([[setting_B_spin]], device=self.device, dtype=torch.float32)
                 
                correction_logits = self.controller.forward(noisy_state_A, noisy_state_B, setting_A_tensor, setting_B_tensor)
                logit_A, logit_B = correction_logits[..., 0], correction_logits[..., 1]

                delayed_logits = self.delay_buffer.push(correction_logits)
                delayed_logit_A, delayed_logit_B = delayed_logits[0], delayed_logits[1]

                substrate_input_A = self._build_input(float(setting_A_bit), delayed_logit_A.item() * actuation_scale)
                substrate_input_B = self._build_input(float(setting_B_bit), delayed_logit_B.item() * actuation_scale)
                self.substrate.step(substrate_input_A, substrate_input_B)

                # Record results for this trial
                y_A = 1 if logit_A.item() >= 0 else -1
                y_B = 1 if logit_B.item() >= 0 else -1
                outputs_A.append(y_A)
                outputs_B.append(y_B)
                settings_log.append((setting_A_bit, setting_B_bit))

        return self._compute_results(outputs_A, outputs_B, settings_log)

    def _compute_results(self, outputs_A, outputs_B, settings_log):
        counts = Counter(settings_log)
        if len(counts) > 0 and len(counts) < 4:
             logging.warning(f"CHSH settings are severely unbalanced: {counts}")
        elif len(counts) == 4 and not all(c == len(settings_log)//4 for c in counts.values()):
            logging.warning(f"CHSH setting counts unbalanced after washout: {counts}")

        S_score, correlations = calculate_chsh_score(outputs_A, outputs_B, settings_log, bootstrap_seed=self.config.get('bootstrap_seed'))
        diagnostics = self._compute_diagnostics(S_score, correlations, outputs_A, outputs_B, settings_log)
        
        return S_score, diagnostics

    def _compute_diagnostics(self, s_score, correlations, outputs_A, outputs_B, settings_log):
        # --- PR-Box Teacher Loss Calculation ---
        teacher_loss = 0.0
        # This block is only active if the config enables it.
        if self.config.get('use_pr_box_teacher', False) and len(settings_log) > 0:
            # Convert {-1, +1} outputs to {0, 1} bits
            o_A_bits = [int((o + 1) / 2) for o in outputs_A]
            o_B_bits = [int((o + 1) / 2) for o in outputs_B]
            
            # Settings are already {0, 1} bits
            s_A_bits = [s[0] for s in settings_log]
            s_B_bits = [s[1] for s in settings_log]

            individual_losses = []
            for i in range(len(settings_log)):
                # Ideal PR-Box rule: o_A XOR o_B == s_A AND s_B
                actual_xor = o_A_bits[i] ^ o_B_bits[i]
                target_and = s_A_bits[i] & s_B_bits[i]
                loss = (actual_xor - target_and)**2  # MSE loss
                individual_losses.append(loss)
            
            teacher_loss = np.mean(individual_losses)

        settings_a_bits = [s[0] for s in settings_log]
        settings_b_bits = [s[1] for s in settings_log]

        delta_ns_A, ci_A = calculate_nonsignaling_metric(outputs_A, settings_a_bits, settings_b_bits, seed=self.config.get('bootstrap_seed', 42))
        delta_ns_B, ci_B = calculate_nonsignaling_metric(outputs_B, settings_b_bits, settings_a_bits, seed=self.config.get('bootstrap_seed', 43)) # Use diff seed
        
        R_cfg = self.config.get('controller_delay', 1.0)
        R_effective = 1.0 / R_cfg if R_cfg >= 1 else round(1.0 / R_cfg)
        
        I_controller_params = sum(p.numel() for p in self.controller.parameters())
        I_substrate = self.config['substrate_params']['N_A'] + self.config['substrate_params']['N_B']
        K_effective = I_controller_params / I_substrate if I_substrate > 0 else 0

        return {
           "S_score": s_score,
           "pr_box_teacher_loss": teacher_loss,
           "correlations": correlations,
           "non_signaling_metric_A": delta_ns_A,
           "non_signaling_metric_A_ci": ci_A,
           "non_signaling_metric_B": delta_ns_B,
           "non_signaling_metric_B_ci": ci_B,
           "R_effective": R_effective,
           "K_effective": K_effective,
        } 