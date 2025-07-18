import torch
import numpy as np
from collections import Counter
import logging

from apsu6.substrate import ClassicalSubstrate
from apsu6.controller import UniversalController
from apsu6.metrics import calculate_chsh_score, calculate_s_score_with_bootstrap, calculate_teacher_loss_gpu, calculate_nonsignaling_metric_gpu
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
        
        # --- STRATEGY 1 MODIFICATION ---
        # We no longer create the substrate and controller here. They will be
        # created on-demand for each evaluation, allowing their parameters
        # to be optimized. We only create them once on the CPU to get shapes
        # and parameter counts, which is much cheaper than a GPU init.
        cpu_device = torch.device('cpu')
        self.temp_controller = UniversalController(**config['controller_params'], device=cpu_device)
        self.anneal_substrate = self.config.get("anneal_substrate", False)
        
        self.chsh_settings = load_chsh_settings(config.get('randomness_file'))

    def get_solution_dimension(self):
        """Returns the total number of parameters to be optimized."""
        dim = sum(p.numel() for p in self.temp_controller.parameters())
        if self.anneal_substrate:
            dim += 4 # sr_A, lr_A, sr_B, lr_B
        return dim

    def _rebuild_components(self, solution_vector):
        """
        Reconstructs the controller and substrate from a flat parameter vector.
        This is the core of Strategy 1: Annealing the Substrate.
        """
        # --- Deconstruct the solution vector ---
        controller_dim = sum(p.numel() for p in self.temp_controller.parameters())
        controller_params_flat = solution_vector[:controller_dim]
        
        # --- Build Substrate ---
        substrate_config = self.config['substrate_params'].copy()
        if self.anneal_substrate:
            # The last 4 parameters are sr_A, lr_A, sr_B, lr_B
            # We clip them to a sensible range.
            substrate_hyperparams = solution_vector[controller_dim:]
            substrate_config['sr_A'] = np.clip(substrate_hyperparams[0], 0.1, 1.5)
            substrate_config['lr_A'] = np.clip(substrate_hyperparams[1], 0.1, 1.0)
            substrate_config['sr_B'] = np.clip(substrate_hyperparams[2], 0.1, 1.5)
            substrate_config['lr_B'] = np.clip(substrate_hyperparams[3], 0.1, 1.0)

        substrate = ClassicalSubstrate(**substrate_config, device=self.device).to(self.device)
        
        # --- Build Controller ---
        controller = UniversalController(**self.config['controller_params'], device=self.device).to(self.device)
        
        # Load weights into the new controller instance
        controller_weights = {}
        start_idx = 0
        target_dtype = next(controller.parameters()).dtype
        for name, param in controller.named_parameters():
            n_params = param.numel()
            p_slice = torch.from_numpy(controller_params_flat[start_idx : start_idx + n_params]).view(param.shape).to(dtype=target_dtype, device=self.device)
            controller_weights[name] = p_slice
            start_idx += n_params
        controller.load_state_dict(controller_weights)
        controller.eval()

        # --- Handle half-precision ---
        if self.config.get('half_precision', False) and self.device.type == 'cuda':
            substrate.half()
            controller.half()
            
        return substrate, controller

    def evaluate_fitness(self, solution_vector: np.ndarray, sensor_noise_std: float = 0.0, num_avg: int = 1) -> tuple[float, dict]:
        """
        Performs one full, sequential fitness evaluation for a given solution vector.
        """
        # 1. Rebuild components based on the solution vector for this evaluation
        substrate, controller = self._rebuild_components(solution_vector)
        
        batch_size = num_avg
        T_total = self.config.get('T_total', 4000)
        washout_steps = self.config.get('washout_steps', 100)

        # Get the correct dtype from the model for all tensor creations
        controller_dtype = next(controller.parameters()).dtype

        # --- Pre-computation and GPU pre-allocation ---
        all_settings_gpu = torch.from_numpy(self.chsh_settings.copy()).long().to(self.device)
        
        # --- Setup Delay Buffer ---
        controller_delay = self.config.get('controller_delay', 1.0)
        delay_buffer = DelayBuffer(
            delay=int(controller_delay) if controller_delay >= 1 else 0, 
            device=self.device,
            dtype=controller_dtype
        )

        # --- Reset stateful components ---
        substrate.reset(batch_size=batch_size)
        delay_buffer.reset(batch_size=batch_size)
        
        # Pre-allocate result tensors on the GPU
        num_scored_steps = T_total - washout_steps
        num_total_results = num_scored_steps * batch_size
        results_outputs_A = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        results_outputs_B = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        results_settings = torch.empty(num_total_results, 2, device=self.device, dtype=torch.long)

        with torch.no_grad():
            for t in range(T_total):
                # 1. Get the current, clean state from the substrate
                state_A, state_B = substrate.get_current_state()
                
                # 2. Apply sensor noise
                if sensor_noise_std > 0:
                    noise_A = torch.randn_like(state_A) * sensor_noise_std
                    noisy_state_A = state_A + noise_A
                    noise_B = torch.randn_like(state_B) * sensor_noise_std
                    noisy_state_B = state_B + noise_B
                else:
                    noisy_state_A = state_A
                    noisy_state_B = state_B
                
                # 3. Get CHSH settings
                setting_A_bit = all_settings_gpu[t, 0].expand(batch_size, 1)
                setting_B_bit = all_settings_gpu[t, 1].expand(batch_size, 1)
                setting_A_spin = bits_to_spins(setting_A_bit).to(controller_dtype)
                setting_B_spin = bits_to_spins(setting_B_bit).to(controller_dtype)
                
                # 4. Compute controller output
                correction_logits, _ = controller.forward(
                    noisy_state_A, noisy_state_B, 
                    setting_A_spin, setting_B_spin
                )

                # 5. Record the CHSH outcome
                if t >= washout_steps:
                    start_idx = (t - washout_steps) * batch_size
                    end_idx = start_idx + batch_size
                    results_outputs_A[start_idx:end_idx] = torch.sign(correction_logits[:, 0])
                    results_outputs_B[start_idx:end_idx] = torch.sign(correction_logits[:, 1])
                    results_settings[start_idx:end_idx] = all_settings_gpu[t].expand(batch_size, 2)
                
                # 6. Apply delay `d` for the substrate's actuation signal
                delayed_logits = delay_buffer.push(correction_logits)
                
                # 7. Evolve substrate
                substrate_dtype = next(substrate.reservoir_A.parameters()).dtype
                substrate_input_A = self._build_input(setting_A_bit, delayed_logits[:, 0].unsqueeze(-1).to(substrate_dtype))
                substrate_input_B = self._build_input(setting_B_bit, delayed_logits[:, 1].unsqueeze(-1).to(substrate_dtype))
                substrate.step(substrate_input_A, substrate_input_B)

        # 8. Scoring
        return self._compute_results(controller, substrate, results_outputs_A, results_outputs_B, results_settings)

    def _build_input(self, setting_bit: torch.Tensor, correction_val: torch.Tensor) -> torch.Tensor:
        """Composes the substrate drive vector [setting_bit, correction_val] for a batch."""
        # Cast setting_bit to the same dtype as correction_val to avoid mixed precision issues.
        return torch.cat((setting_bit.to(correction_val.dtype), correction_val), dim=-1)

    def _compute_results(self, controller, substrate, outputs_A_gpu: torch.Tensor, outputs_B_gpu: torch.Tensor, settings_log_gpu: torch.Tensor) -> tuple[float, dict]:
        # The S-score is calculated directly on the GPU.
        S_score_gpu, correlations_gpu, s_ci_bounds, p_classical, p_tsirelson = calculate_s_score_with_bootstrap(
            outputs_A_gpu, 
            outputs_B_gpu, 
            settings_log_gpu,
            seed=self.config.get('bootstrap_seed', 42)
        )
        
        # Teacher loss calculation on GPU
        teacher_loss_gpu = torch.tensor(0.0, device=self.device)
        if self.config.get('use_pr_box_teacher', False):
             teacher_loss_gpu = calculate_teacher_loss_gpu(outputs_A_gpu, outputs_B_gpu, settings_log_gpu)

        # Non-signaling metric calculation on GPU
        settings_a_gpu = settings_log_gpu[:, 0]
        settings_b_gpu = settings_log_gpu[:, 1]
        delta_ns_A_gpu, ci_A_gpu = calculate_nonsignaling_metric_gpu(outputs_A_gpu, settings_a_gpu, settings_b_gpu, seed=self.config.get('bootstrap_seed', 42))
        delta_ns_B_gpu, ci_B_gpu = calculate_nonsignaling_metric_gpu(outputs_B_gpu, settings_b_gpu, settings_a_gpu, seed=self.config.get('bootstrap_seed', 43))
        
        # --- Final CPU Transfer ---
        # Only transfer the final scalar results back to the CPU.
        diagnostics = {
           "S_score": S_score_gpu.item(),
           "S_score_ci": (s_ci_bounds[0].item(), s_ci_bounds[1].item()),
           "p_classical": p_classical.item(),
           "p_tsirelson": p_tsirelson.item(),
           "pr_box_teacher_loss": teacher_loss_gpu.item(),
           "correlations": {k: v.item() for k, v in correlations_gpu.items()},
           "non_signaling_metric_A": delta_ns_A_gpu.item(),
           "non_signaling_metric_A_ci": (ci_A_gpu[0].item(), ci_A_gpu[1].item()),
           "non_signaling_metric_B": delta_ns_B_gpu.item(),
           "non_signaling_metric_B_ci": (ci_B_gpu[0].item(), ci_B_gpu[1].item()),
           "R_effective": 1.0 / self.config.get('controller_delay', 1.0) if self.config.get('controller_delay', 1.0) >= 1 else round(1.0 / self.config.get('controller_delay', 1.0)),
           "K_effective": sum(p.numel() for p in controller.parameters()) / (substrate.N_A + substrate.N_B),
        }
        
        return diagnostics['S_score'], diagnostics


    def _compute_diagnostics(self, s_score, correlations, outputs_A, outputs_B, settings_log):
        # This entire method is now deprecated in favor of _compute_results on the GPU.
        # It is kept here for reference and potential CPU-based debugging.
        pass 