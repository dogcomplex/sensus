import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import logging
from typing import List, Tuple, Optional

from apsu6.substrate import ClassicalSubstrate
from apsu6.controller import UniversalController
from apsu6.metrics import calculate_chsh_score, calculate_nonsignaling_metric
from apsu6.utils import load_chsh_settings, bits_to_spins


# @torch.jit.script # Decorator is removed from class definition
class CausalLoopJIT(nn.Module):
    """
    A JIT-compiled module to run the entire causal simulation loop efficiently on the GPU.
    """
    def __init__(self, substrate: ClassicalSubstrate, controller: UniversalController, delay_buffer: 'DelayBuffer'):
        super().__init__()
        self.substrate = substrate
        self.controller = controller
        self.delay_buffer = delay_buffer

    def forward(self, settings_seq: torch.Tensor, noise_seq_A: torch.Tensor, noise_seq_B: torch.Tensor, 
                T_total: int, washout_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = settings_seq.shape[0]
        
        self.substrate.reset(batch_size)
        self.delay_buffer.reset(batch_size)
        
        outputs_A: List[torch.Tensor] = []
        outputs_B: List[torch.Tensor] = []
        logged_settings: List[torch.Tensor] = []

        for t in range(T_total):
            state_A, state_B = self.substrate.get_current_state()
            
            noisy_state_A = state_A + noise_seq_A[:, t, :]
            noisy_state_B = state_B + noise_seq_B[:, t, :]
            
            setting_A_bit = settings_seq[:, t, 0].unsqueeze(-1)
            setting_B_bit = settings_seq[:, t, 1].unsqueeze(-1)
            setting_A_spin = bits_to_spins(setting_A_bit)
            setting_B_spin = bits_to_spins(setting_B_bit)
            
            R_speed = self.controller.R
            thought_h_state: torch.Tensor | None = None
            correction_logits: torch.Tensor | None = None

            if R_speed > 1.0:
                internal_iterations = int(round(R_speed))
                for _ in range(internal_iterations):
                    correction_logits, thought_h_state = self.controller.forward(
                        noisy_state_A, noisy_state_B, setting_A_spin, setting_B_spin, thought_h_in=thought_h_state
                    )
            else:
                correction_logits, _ = self.controller.forward(
                    noisy_state_A, noisy_state_B, setting_A_spin, setting_B_spin
                )
            
            # This assert is essential for the JIT compiler to resolve types
            assert correction_logits is not None

            if t >= washout_steps:
                outputs_A.append(torch.sign(correction_logits[:, 0]))
                outputs_B.append(torch.sign(correction_logits[:, 1]))
                logged_settings.append(torch.cat([setting_A_bit, setting_B_bit], dim=-1))
            
            # The DelayBuffer now expects a 3D tensor: [1, Batch, Channels]
            delayed_logits = self.delay_buffer(correction_logits.unsqueeze(0)) 
            
            substrate_input_A = torch.cat((setting_A_bit.float(), delayed_logits[:, 0].unsqueeze(-1)), dim=-1)
            substrate_input_B = torch.cat((setting_B_bit.float(), delayed_logits[:, 1].unsqueeze(-1)), dim=-1)
            self.substrate.step(substrate_input_A, substrate_input_B)

        return torch.stack(outputs_A, dim=1), torch.stack(outputs_B, dim=1), torch.stack(logged_settings, dim=1)


class DelayBuffer(nn.Module):
    """Handles the delay `d` for the controller's correction signals for a batch."""
    buffer: Optional[torch.Tensor]

    def __init__(self, delay: int, num_channels: int = 2, device: torch.device = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__()
        self.delay = int(delay)
        if self.delay < 0:
            raise ValueError("Delay cannot be negative.")
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.buffer = None
    
    def forward(self, signal_batch: torch.Tensor) -> torch.Tensor:
        """Pushes a new batch of signals in, returns the oldest batch."""
        if self.delay == 0:
            return signal_batch
        
        buffer = self.buffer
        if buffer is None:
             raise RuntimeError("Buffer must be reset with a batch size before use.")
        
        # JIT-compatible circular buffer update. This pattern avoids in-place
        # assignment to a slice, which is not supported by the JIT compiler.
        oldest_signal = buffer[0, :, :].clone()

        # Reshape signal from (B, C) to (1, B, C) for concatenation
        reshaped_signal = signal_batch.unsqueeze(0)
        
        new_buffer_part = buffer[1:]
        self.buffer = torch.cat((new_buffer_part, reshaped_signal), dim=0)
        
        return oldest_signal
        
    def reset(self, batch_size: int = 1):
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
        
        controller_delay = config.get('controller_delay', 1.0)
        self.delay_buffer = DelayBuffer(
            delay=int(controller_delay) if controller_delay >= 1 else 0, 
            device=self.device
        )
        
        self.chsh_settings = load_chsh_settings(config.get('randomness_file'))
        self.noise_rng = torch.Generator(device=self.device)
        self.noise_rng.manual_seed(config.get('noise_seed', 42))

    def evaluate_fitness(self, controller_weights: dict, sensor_noise_std: float = 0.0, num_avg: int = 1) -> tuple[float, dict]:
        """
        Performs one full, JIT-compiled fitness evaluation.
        """
        batch_size = num_avg
        self.controller.load_state_dict(controller_weights)
        self.controller.eval()
        
        T_total = self.config.get('T_total', 4000)
        washout_steps = self.config.get('washout_steps', 100)
        
        # Pre-generate all data for the JIT module
        # Use .copy() to avoid the non-writable numpy array warning
        settings_seq = torch.from_numpy(self.chsh_settings.copy()).long().to(self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        noise_shape = (batch_size, T_total, self.substrate.N_A)
        noise_seq_A = torch.randn(noise_shape, device=self.device, generator=self.noise_rng) * sensor_noise_std
        noise_seq_B = torch.randn(noise_shape, device=self.device, generator=self.noise_rng) * sensor_noise_std

        # Instantiate the module, then script the instance
        causal_loop_module = CausalLoopJIT(self.substrate, self.controller, self.delay_buffer)
        causal_loop = torch.jit.script(causal_loop_module)

        with torch.no_grad():
            outputs_A_seq, outputs_B_seq, settings_log_seq = causal_loop(
                settings_seq, noise_seq_A, noise_seq_B, T_total, washout_steps
            )

        # Flatten results for scoring
        final_outputs_A = outputs_A_seq.reshape(-1)
        final_outputs_B = outputs_B_seq.reshape(-1)
        final_settings_log = settings_log_seq.reshape(-1, 2)

        return self._compute_results(final_outputs_A, final_outputs_B, final_settings_log)

    def _build_input(self, setting_bit: torch.Tensor, correction_val: torch.Tensor) -> torch.Tensor:
        """Composes the substrate drive vector [setting_bit, correction_val] for a batch."""
        return torch.cat((setting_bit.float(), correction_val), dim=-1)

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