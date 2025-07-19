import torch
import numpy as np
from collections import Counter
import logging
import torch.nn as nn

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
        
        # --- STRATEGY 3 MODIFICATION ---
        # Add a separate, powerful readout model for curriculum learning.
        # N_A + N_B -> 2 outputs (A and B)
        substrate_dim = config['substrate_params']['N_A'] + config['substrate_params']['N_B']
        self.post_hoc_readout = nn.Linear(substrate_dim, 2).to(self.device)
        
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
            # "CONSTRAINED CHAOS": Clip to a narrow, computationally rich range.
            substrate_hyperparams = solution_vector[controller_dim:]
            substrate_config['sr_A'] = np.clip(substrate_hyperparams[0], 0.9, 1.2)
            substrate_config['lr_A'] = np.clip(substrate_hyperparams[1], 0.1, 0.4)
            substrate_config['sr_B'] = np.clip(substrate_hyperparams[2], 0.9, 1.2)
            substrate_config['lr_B'] = np.clip(substrate_hyperparams[3], 0.1, 0.4)

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
        controller.reset() # Reset the internal state (e.g., GRU hidden state)

        # --- Handle half-precision ---
        if self.config.get('half_precision', False) and self.device.type == 'cuda':
            substrate.half()
            controller.half()
            
        return substrate, controller

    def _run_simulation_pass(self, substrate, controller):
        """
        Runs a single full pass of the simulation in 'end_to_end' mode.
        """
        batch_size = self.config['evaluation']['num_avg']
        T_total = self.config.get('T_total', 4000)
        washout_steps = self.config.get('washout_steps', 100)
        controller_dtype = next(controller.parameters()).dtype

        all_settings_gpu = torch.from_numpy(self.chsh_settings.copy()).long().to(self.device)
        delay_buffer = DelayBuffer(delay=int(self.config.get('controller_delay', 1.0)), device=self.device, dtype=controller_dtype)

        substrate.reset(batch_size=batch_size)
        controller.reset()
        delay_buffer.reset(batch_size=batch_size)
        
        num_scored_steps = T_total - washout_steps
        num_total_results = num_scored_steps * batch_size
        results_outputs_A = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        results_outputs_B = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
        results_settings = torch.empty(num_total_results, 2, device=self.device, dtype=torch.long)
        
        h_state = None
        with torch.no_grad():
            for t in range(T_total):
                state_A, state_B = substrate.get_current_state()
                
                # For simplicity, sensor noise is omitted in this refactor. Can be added back.
                noisy_state_A, noisy_state_B = state_A, state_B
                
                setting_A_bit = all_settings_gpu[t, 0].expand(batch_size, 1)
                setting_B_bit = all_settings_gpu[t, 1].expand(batch_size, 1)
                setting_A_spin = bits_to_spins(setting_A_bit).to(controller_dtype)
                setting_B_spin = bits_to_spins(setting_B_bit).to(controller_dtype)
                
                correction_logits, h_state = controller.forward(noisy_state_A, noisy_state_B, setting_A_spin, setting_B_spin, h_prev=h_state)

                if t >= washout_steps:
                    start_idx = (t - washout_steps) * batch_size
                    end_idx = start_idx + batch_size
                    results_settings[start_idx:end_idx] = all_settings_gpu[t].expand(batch_size, 2)
                    results_outputs_A[start_idx:end_idx] = torch.sign(correction_logits[:, 0])
                    results_outputs_B[start_idx:end_idx] = torch.sign(correction_logits[:, 1])

                delayed_logits = delay_buffer.push(correction_logits)
                substrate_dtype = next(substrate.reservoir_A.parameters()).dtype
                substrate_input_A = self._build_input(setting_A_bit, delayed_logits[:, 0].unsqueeze(-1).to(substrate_dtype))
                substrate_input_B = self._build_input(setting_B_bit, delayed_logits[:, 1].unsqueeze(-1).to(substrate_dtype))
                substrate.step(substrate_input_A, substrate_input_B)

        return results_outputs_A, results_outputs_B, results_settings


    def evaluate_fitness(self, solution_vector: np.ndarray) -> tuple[float, dict]:
        """
        Performs one full, sequential fitness evaluation for a given solution vector.
        The readout_mode is now fixed to 'end_to_end'.
        """
        substrate, controller = self._rebuild_components(solution_vector)
        
        outputs_A, outputs_B, settings = self._run_simulation_pass(substrate, controller)
        
        return self._compute_results(controller, substrate, outputs_A, outputs_B, settings.view(-1, 2))

    def _build_input(self, setting_bit: torch.Tensor, correction_val: torch.Tensor) -> torch.Tensor:
        """Composes the substrate drive vector [setting_bit, correction_val] for a batch."""
        # Cast setting_bit to the same dtype as correction_val to avoid mixed precision issues.
        return torch.cat((setting_bit.to(correction_val.dtype), correction_val), dim=-1)

    def _compute_results(self, controller, substrate, outputs_A_gpu: torch.Tensor, outputs_B_gpu: torch.Tensor, settings_log_gpu: torch.Tensor) -> tuple[float, dict]:
        # --- Memory Profiling ---
        if self.device.type == 'cuda':
            # Clear cache to get a more accurate reading of persistent memory
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            logging.debug(f"GPU Memory at results: {mem_allocated:.2f}MB allocated / {mem_reserved:.2f}MB reserved")

        # The S-score is calculated directly on the GPU.
        S_score_gpu, correlations_gpu, s_ci_bounds, p_classical, p_tsirelson = calculate_s_score_with_bootstrap(
            outputs_A_gpu, 
            outputs_B_gpu, 
            settings_log_gpu,
            seed=self.config.get('bootstrap_seed', 42),
            use_cpu_fallback=self.config['evaluation'].get('use_cpu_fallback_for_metrics', False)
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
        # CRITICAL: Convert all tensors to plain Python numbers/lists to prevent
        # retaining computation graphs in memory across processes.
        diagnostics = {
           "S_score": S_score_gpu.item(),
           "S_score_ci": s_ci_bounds.tolist(),
           "p_classical": p_classical.item(),
           "p_tsirelson": p_tsirelson.item(),
           "pr_box_teacher_loss": teacher_loss_gpu.item(),
           "correlations": {str(k): v.item() for k, v in correlations_gpu.items()},
           "non_signaling_metric_A": delta_ns_A_gpu.item(),
           "non_signaling_metric_A_ci": ci_A_gpu.tolist(),
           "non_signaling_metric_B": delta_ns_B_gpu.item(),
           "non_signaling_metric_B_ci": ci_B_gpu.tolist(),
           "R_effective": 1.0 / self.config.get('controller_delay', 1.0) if self.config.get('controller_delay', 1.0) >= 1 else round(1.0 / self.config.get('controller_delay', 1.0)),
           "K_effective": sum(p.numel() for p in controller.parameters()) / (substrate.N_A + substrate.N_B),
        }
        
        return diagnostics['S_score'], diagnostics


    def _compute_diagnostics(self, s_score, correlations, outputs_A, outputs_B, settings_log):
        # This entire method is now deprecated in favor of _compute_results on the GPU.
        # It is kept here for reference and potential CPU-based debugging.
        pass 