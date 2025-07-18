import torch
from pathlib import Path
import numpy as np
from collections import Counter
import logging

from apsu6.substrate import ClassicalSubstrate
from apsu6.controller import UniversalController
from apsu6.metrics import calculate_chsh_score, calculate_teacher_loss_gpu, calculate_nonsignaling_metric_gpu
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
        self.noise_rng = torch.Generator(device=self.device)
        self.noise_rng.manual_seed(config.get('noise_seed', 42))

 
    def evaluate_fitness(self, controller, current_lambda=0.0, current_noise=0.0):
        """
        Performs one full fitness evaluation for a given controller instance.
        This is the function that the optimizer's parallel workers will call repeatedly.
        """
        batch_size = self.config['evaluation']['num_avg']
        T_total = self.config.get('T_total', 4000)
        washout_steps = self.config.get('washout_steps', 100)

        # Get the correct dtype from the model for all tensor creations
        controller_dtype = next(controller.parameters()).dtype

        # --- Pre-computation and GPU pre-allocation ---
        # 1. Pre-transfer all CHSH settings to the GPU at once.
        all_settings_gpu = torch.from_numpy(self.chsh_settings.copy()).long().to(self.device)
        
        # --- Reset stateful components ---
        self.substrate.reset(batch_size=batch_size)
        # The controller is passed in already configured with weights, so we just set to eval mode.
        controller.eval()
        self.delay_buffer.reset(batch_size=batch_size)
        
        # Pre-allocate result tensors on the GPU to avoid CPU sync in the loop
        num_scored_steps = T_total - washout_steps
        num_total_results = num_scored_steps * batch_size
        results_outputs_A = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        results_outputs_B = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        results_settings = torch.empty(num_total_results, 2, device=self.device, dtype=torch.long)
        
        all_controller_logits_A = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        all_controller_logits_B = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        all_teacher_logits_A = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        all_teacher_logits_B = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)


        with torch.no_grad():
            for t in range(T_total):
                # 1. Get the current, clean state from the substrate
                state_A, state_B = self.substrate.get_current_state()
                
                # 2. Apply sensor noise on-demand, inside the loop.
                if current_noise > 0:
                    noise_A = torch.randn_like(state_A) * current_noise
                    noisy_state_A = state_A + noise_A
                    noise_B = torch.randn_like(state_B) * current_noise
                    noisy_state_B = state_B + noise_B
                else:
                    noisy_state_A = state_A
                    noisy_state_B = state_B
                
                # 3. Get CHSH settings by slicing the pre-allocated GPU tensor
                setting_A_bit = all_settings_gpu[t, 0].expand(batch_size, 1)
                setting_B_bit = all_settings_gpu[t, 1].expand(batch_size, 1)
                setting_A_spin = bits_to_spins(setting_A_bit).to(controller_dtype)
                setting_B_spin = bits_to_spins(setting_B_bit).to(controller_dtype)
                
                # 4. Compute controller output.
                correction_logits, _ = controller.forward(
                    noisy_state_A, noisy_state_B, 
                    setting_A_spin, setting_B_spin
                )

                # 5. Record the CHSH outcome for this step *before* any delay
                if t >= washout_steps:
                    start_idx = (t - washout_steps) * batch_size
                    end_idx = start_idx + batch_size
                    results_outputs_A[start_idx:end_idx] = torch.sign(correction_logits[:, 0])
                    results_outputs_B[start_idx:end_idx] = torch.sign(correction_logits[:, 1])
                    results_settings[start_idx:end_idx] = all_settings_gpu[t].expand(batch_size, 2)

                    if self.config.get('use_pr_box_teacher', False):
                        teacher_A, teacher_B = self._get_pr_box_teacher_output(
                            setting_A_bit.squeeze(-1), setting_B_bit.squeeze(-1)
                        )
                        all_controller_logits_A[start_idx:end_idx] = correction_logits[:, 0]
                        all_controller_logits_B[start_idx:end_idx] = correction_logits[:, 1]
                        all_teacher_logits_A[start_idx:end_idx] = teacher_A
                        all_teacher_logits_B[start_idx:end_idx] = teacher_B

                # 6. Apply delay `d` for the substrate's actuation signal
                delayed_logits = self.delay_buffer.push(correction_logits)
                
                # 7. Evolve substrate using the (potentially delayed) correction
                # Ensure input to substrate matches its precision. It might be different from controller.
                substrate_dtype = next(self.substrate.reservoir_A.parameters()).dtype
                substrate_input_A = self._build_input(setting_A_bit, delayed_logits[:, 0].unsqueeze(-1).to(substrate_dtype))
                substrate_input_B = self._build_input(setting_B_bit, delayed_logits[:, 1].unsqueeze(-1).to(substrate_dtype))
                self.substrate.step(substrate_input_A, substrate_input_B)

        # 8. Scoring (after the loop)
        return self._compute_results(
            results_outputs_A, results_outputs_B, results_settings,
            all_controller_logits_A, all_controller_logits_B,
            all_teacher_logits_A, all_teacher_logits_B
        )

    def _build_input(self, setting_bit: torch.Tensor, correction_val: torch.Tensor) -> torch.Tensor:
        """Composes the substrate drive vector [setting_bit, correction_val] for a batch."""
        return torch.cat((setting_bit.to(correction_val.dtype), correction_val), dim=-1)

    def _get_pr_box_teacher_output(self, setting_A_bit, setting_B_bit):
        """Calculates the ideal PR-Box output for teaching."""
        # PR-Box rule: outcome = setting_a AND setting_b
        pr_box_logic = (setting_A_bit * setting_B_bit).float() # 0 if either is 0, 1 if both are 1
        # Convert to spin {-1, +1} domain for correlation
        y = 1.0 - 2.0 * pr_box_logic
        # For the teacher, we assume perfect correlation, so y_A = y_B
        return y, y

    def _compute_results(self, outputs_A_gpu, outputs_B_gpu, settings_log_gpu,
                         controller_A_logits, controller_B_logits,
                         teacher_A_logits, teacher_B_logits):
        S_score_gpu, correlations_gpu = calculate_chsh_score(outputs_A_gpu, outputs_B_gpu, settings_log_gpu)
        
        teacher_loss_gpu = torch.tensor(0.0, device=self.device)
        if self.config.get('use_pr_box_teacher', False):
             teacher_loss_gpu = calculate_teacher_loss_gpu(
                 controller_A_logits, controller_B_logits,
                 teacher_A_logits, teacher_B_logits
            )

        settings_a_gpu = settings_log_gpu[:, 0]
        settings_b_gpu = settings_log_gpu[:, 1]
        delta_ns_A_gpu, ci_A_gpu = calculate_nonsignaling_metric_gpu(outputs_A_gpu, settings_a_gpu, settings_b_gpu, seed=self.config.get('bootstrap_seed', 42))
        delta_ns_B_gpu, ci_B_gpu = calculate_nonsignaling_metric_gpu(outputs_B_gpu, settings_b_gpu, settings_a_gpu, seed=self.config.get('bootstrap_seed', 43))
        
        diagnostics = {
           "S_score": S_score_gpu.item(),
           "teacher_loss": teacher_loss_gpu.item(),
           "correlations": {k: v.item() for k, v in correlations_gpu.items()},
           "non_signaling_metric_A": delta_ns_A_gpu.item(),
           "non_signaling_metric_A_ci": (ci_A_gpu[0].item(), ci_A_gpu[1].item()),
           "non_signaling_metric_B": delta_ns_B_gpu.item(),
           "non_signaling_metric_B_ci": (ci_B_gpu[0].item(), ci_B_gpu[1].item()),
        }
        
        return diagnostics['S_score'], diagnostics