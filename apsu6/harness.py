import torch
import numpy as np
from collections import Counter
import logging

from apsu6.substrate import ClassicalSubstrate
from apsu6.controller import UniversalController
from apsu6.metrics import calculate_chsh_score, calculate_nonsignaling_metric
from apsu6.utils import load_chsh_settings, bits_to_spins

class DelayBuffer:
    """Handles the delay `d` for the controller's correction signals for a batch."""
    def __init__(self, delay: int, num_channels: int = 2, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        self.delay = int(delay)
        if self.delay < 0:
            raise ValueError("Delay cannot be negative.")
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.buffer = None
    
    def push(self, signal_batch: torch.Tensor) -> torch.Tensor:
        """Pushes a new batch of signals in, returns the oldest batch."""
        if self.delay == 0:
            return signal_batch
        
        if self.buffer is None:
             raise RuntimeError("Buffer must be reset with a batch size before use.")
        
        assert signal_batch.shape[0] == self.buffer.shape[1], "Signal batch size must match buffer's."

        oldest_signal = self.buffer[0, :, :].clone()
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        self.buffer[-1, :, :] = signal_batch
        return oldest_signal
        
    def reset(self, batch_size=1):
        """Resets the buffer for a new batch size."""
        self.buffer = torch.zeros(self.delay, batch_size, self.num_channels, device=self.device, dtype=self.dtype)

class ExperimentHarness:
    """
    The main orchestration script. Sets up the experiment, runs the simulation 
    loop, gathers data, and computes the final fitness score.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Harness selected device: {self.device}")
        
        self.substrate = ClassicalSubstrate(**config['substrate_params'], device=self.device).to(self.device)
        self.controller = UniversalController(**config['controller_params'], device=self.device).to(self.device)
        
        # --- Handle half-precision ---
        if config.get('half_precision', False) and self.device.type == 'cuda':
            logging.info("Using half-precision (float16) for accelerated training.")
            self.substrate.half()
            self.controller.half()
        
        controller_delay = config.get('controller_delay', 1.0)
        # The buffer must match the precision of the controller that feeds it.
        buffer_dtype = next(self.controller.parameters()).dtype
        self.delay_buffer = DelayBuffer(
            delay=int(controller_delay) if controller_delay >= 1 else 0, 
            device=self.device,
            dtype=buffer_dtype
        )
        
        self.chsh_settings = load_chsh_settings(config.get('randomness_file'))

    def evaluate_fitness(self, controller_weights: dict, sensor_noise_std: float = 0.0, num_avg: int = 1) -> tuple[float, dict]:
        """
        Performs one full, sequential fitness evaluation for a given set of controller weights.
        This version correctly models the step-by-step evolution of the system to prevent
        causality violations.
        """
        batch_size = num_avg
        T_total = self.config.get('T_total', 4000)
        washout_steps = self.config.get('washout_steps', 100)

        # --- Pre-computation and GPU pre-allocation ---
        # 1. Pre-transfer all CHSH settings to the GPU at once.
        all_settings_gpu = torch.from_numpy(self.chsh_settings.copy()).long().to(self.device)
        
        # 2. Pre-generate all sensor noise required for the entire simulation at once.
        noise_shape_A = (T_total, batch_size, self.substrate.N_A)
        noise_shape_B = (T_total, batch_size, self.substrate.N_B)
        buffer_dtype = next(self.controller.parameters()).dtype
        # Use the default, per-worker-seeded RNG.
        all_noise_A_gpu = torch.randn(noise_shape_A, device=self.device).to(buffer_dtype) * sensor_noise_std
        all_noise_B_gpu = torch.randn(noise_shape_B, device=self.device).to(buffer_dtype) * sensor_noise_std
        
        # --- Reset stateful components ---
        self.substrate.reset(batch_size=batch_size)
        self.controller.load_state_dict(controller_weights)
        self.controller.eval()
        self.delay_buffer.reset(batch_size=batch_size)

        # Get the correct dtype from the model, in case we are in half-precision mode
        controller_dtype = next(self.controller.parameters()).dtype

        outputs_A, outputs_B, settings_log = [], [], []

        with torch.no_grad():
            for t in range(T_total):
                # 1. Get the current, clean state from the substrate
                state_A, state_B = self.substrate.get_current_state()
                
                # 2. Apply pre-generated sensor noise by slicing the GPU tensor
                noisy_state_A = state_A + all_noise_A_gpu[t]
                noisy_state_B = state_B + all_noise_B_gpu[t]
                
                # 3. Get CHSH settings by slicing the pre-allocated GPU tensor
                setting_A_bit = all_settings_gpu[t, 0].expand(batch_size, 1)
                setting_B_bit = all_settings_gpu[t, 1].expand(batch_size, 1)
                # Ensure the spin tensors match the controller's precision
                setting_A_spin = bits_to_spins(setting_A_bit).to(controller_dtype)
                setting_B_spin = bits_to_spins(setting_B_bit).to(controller_dtype)
                
                # 4. Compute controller output. The controller now handles the "thinking loop" internally.
                correction_logits, _ = self.controller.forward(
                    noisy_state_A, noisy_state_B, 
                    setting_A_spin, setting_B_spin
                )

                # 5. Record the CHSH outcome for this step *before* any delay
                if t >= washout_steps:
                    # Binarize and store results for each item in the batch
                    y_A = torch.sign(correction_logits[:, 0]).cpu().tolist()
                    y_B = torch.sign(correction_logits[:, 1]).cpu().tolist()
                    outputs_A.extend(y_A)
                    outputs_B.extend(y_B)
                    # Log the settings for each item in the batch
                    settings_for_step = all_settings_gpu[t].expand(batch_size, 2).cpu().numpy()
                    settings_log.extend(map(tuple, settings_for_step))
                
                # 6. Apply delay `d` for the substrate's actuation signal
                delayed_logits = self.delay_buffer.push(correction_logits)
                
                # 7. Evolve substrate using the (potentially delayed) correction
                # Ensure input to substrate matches its precision. It might be different from controller.
                substrate_dtype = next(self.substrate.reservoir_A.parameters()).dtype
                substrate_input_A = self._build_input(setting_A_bit, delayed_logits[:, 0].unsqueeze(-1).to(substrate_dtype))
                substrate_input_B = self._build_input(setting_B_bit, delayed_logits[:, 1].unsqueeze(-1).to(substrate_dtype))
                self.substrate.step(substrate_input_A, substrate_input_B)

        # 8. Scoring (after the loop)
        # Convert collected lists to GPU tensors for fast scoring
        final_outputs_A = torch.tensor(outputs_A, device=self.device, dtype=torch.float32)
        final_outputs_B = torch.tensor(outputs_B, device=self.device, dtype=torch.float32)
        final_settings_log = torch.tensor(settings_log, device=self.device, dtype=torch.long)

        return self._compute_results(final_outputs_A, final_outputs_B, final_settings_log)

    def _build_input(self, setting_bit: torch.Tensor, correction_val: torch.Tensor) -> torch.Tensor:
        """Composes the substrate drive vector [setting_bit, correction_val] for a batch."""
        # Cast setting_bit to the same dtype as correction_val to avoid mixed precision issues.
        return torch.cat((setting_bit.to(correction_val.dtype), correction_val), dim=-1)

    def _compute_results(self, outputs_A_gpu: torch.Tensor, outputs_B_gpu: torch.Tensor, settings_log_gpu: torch.Tensor) -> tuple[float, dict]:
        # The S-score is calculated directly on the GPU.
        S_score_gpu, correlations_gpu = calculate_chsh_score(outputs_A_gpu, outputs_B_gpu, settings_log_gpu)
        
        # Convert to CPU for diagnostics and optimizer, which expect floats/dicts
        S_score_cpu = S_score_gpu.item()
        correlations_cpu = {k: v.item() for k, v in correlations_gpu.items()}
        
        # The non-signaling metric is less performance-critical and can remain on CPU for now.
        diagnostics = self._compute_diagnostics(
            S_score_cpu, 
            correlations_cpu, 
            outputs_A_gpu.cpu().tolist(), 
            outputs_B_gpu.cpu().tolist(), 
            settings_log_gpu.cpu().tolist()
        )
        
        return S_score_cpu, diagnostics

    def _compute_diagnostics(self, s_score, correlations, outputs_A, outputs_B, settings_log):
        # Teacher loss calculation
        teacher_loss = 0.0
        if self.config.get('use_pr_box_teacher', False) and len(settings_log) > 0:
            o_A_bits = (np.array(outputs_A) + 1) / 2
            o_B_bits = (np.array(outputs_B) + 1) / 2
            s_A_bits = np.array(settings_log)[:, 0]
            s_B_bits = np.array(settings_log)[:, 1]

            actual_xor = np.bitwise_xor(o_A_bits.astype(int), o_B_bits.astype(int))
            target_and = np.bitwise_and(s_A_bits.astype(int), s_B_bits.astype(int))
            
            teacher_loss = np.mean((actual_xor - target_and)**2)

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