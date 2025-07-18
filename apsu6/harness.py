import torch
import numpy as np
from collections import Counter
import logging
import torch.nn as nn

from apsu6.pytorch_substrate import ClassicalSubstrate
from apsu6.controller import UniversalController
from apsu6.metrics import calculate_s_score_with_bootstrap, calculate_teacher_loss_gpu, calculate_nonsignaling_metric_gpu
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
    ARCHITECTURAL FIX: This harness now creates persistent models once and loads
    weights on each call, avoiding catastrophic memory churn.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Harness selected device: {self.device}")
        
        # --- ARCHITECTURAL FIX ---
        # Create persistent substrate and controller instances ONCE.
        self.substrate = ClassicalSubstrate(**config['substrate_params'], device=self.device)
        self.controller = UniversalController(**config['controller_params'], device=self.device)
        self.anneal_substrate = self.config.get("anneal_substrate", False)
        
        self.chsh_settings = load_chsh_settings(config.get('randomness_file'))

    def get_solution_dimension(self):
        """Returns the total number of parameters to be optimized."""
        dim = sum(p.numel() for p in self.controller.parameters())
        if self.anneal_substrate:
            dim += sum(p.numel() for p in self.substrate.parameters())
            dim += 4 # sr_A, lr_A, sr_B, lr_B for reconstruction, not direct optimization
        return dim

    def _load_solution_vector(self, solution_vector):
        """
        Loads a flat parameter vector into the persistent controller and substrate.
        This is the new, lightweight replacement for _rebuild_components.
        """
        # --- Deconstruct the solution vector ---
        controller_dim = sum(p.numel() for p in self.controller.parameters())
        controller_params_flat = solution_vector[:controller_dim]

        # --- Load Controller Weights ---
        start_idx = 0
        target_dtype = next(self.controller.parameters()).dtype
        for param in self.controller.parameters():
            n_params = param.numel()
            p_slice = torch.from_numpy(controller_params_flat[start_idx : start_idx + n_params]).view(param.shape)
            param.data.copy_(p_slice.to(param.device, dtype=target_dtype))
            start_idx += n_params

        # --- Load Substrate Weights and Hyperparameters (if annealing) ---
        if self.anneal_substrate:
            substrate_params_dim = sum(p.numel() for p in self.substrate.parameters())
            substrate_params_flat = solution_vector[controller_dim : controller_dim + substrate_params_dim]
            
            start_idx = 0
            target_dtype = next(self.substrate.parameters()).dtype
            for param in self.substrate.parameters():
                n_params = param.numel()
                p_slice = torch.from_numpy(substrate_params_flat[start_idx : start_idx + n_params]).view(param.shape)
                param.data.copy_(p_slice.to(param.device, dtype=target_dtype))
                start_idx += n_params

            # The 4 hyperparameters are not part of the model's state_dict,
            # but are used to reconstruct it if needed. Here, we just acknowledge them.
            # The actual substrate instance is not rebuilt, just its weights are updated.
            
    def evaluate_fitness(self, solution_vector: np.ndarray, sensor_noise_std: float = 0.0, num_avg: int = 1, readout_mode: str = "end_to_end") -> tuple[float, dict]:
        """
        Performs one full, sequential fitness evaluation for a given solution vector.
        
        Args:
            solution_vector (np.ndarray): The flat parameter vector for the controller.
            sensor_noise_std (float): Standard deviation of sensor noise.
            num_avg (int): Number of steps to average over for the fitness score.
            readout_mode (str): 'end_to_end' (default) or 'post_hoc'.
        """
        # 1. Load weights into the persistent models. This is now a very cheap operation.
        self._load_solution_vector(solution_vector)
        
        # Get references to the persistent models for clarity
        substrate, controller = self.substrate, self.controller
        controller.eval()
        controller.reset() # Reset the internal state (e.g., GRU hidden state)

        # --- Handle half-precision ---
        if self.config.get('half_precision', False) and self.device.type == 'cuda':
            substrate.half()
            controller.half()
            
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
        
        # Pre-allocate result tensors on the GPU ONLY if needed
        if readout_mode == 'end_to_end':
            num_scored_steps = T_total - washout_steps
            num_total_results = num_scored_steps * batch_size
            results_outputs_A = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
            results_outputs_B = torch.empty(num_total_results, device=self.device, dtype=controller_dtype)
            results_settings = torch.empty(num_total_results, 2, device=self.device, dtype=torch.long)
        else: # For post_hoc, we just need placeholders for the final assignment
            results_outputs_A, results_outputs_B, results_settings = None, None, None
        
        # --- Memory-Efficient State Storage for Post-Hoc Readout ---
        if readout_mode == 'post_hoc':
            post_hoc_batch_size = self.config.get('post_hoc_batch_size', 1024)
            substrate_dtype = next(substrate.parameters()).dtype
            # Use reservoir sampling to collect a random subset of states
            stored_states = torch.empty(post_hoc_batch_size, substrate.N_A + substrate.N_B, device=self.device, dtype=substrate_dtype)
            stored_settings = torch.empty(post_hoc_batch_size, 2, device=self.device, dtype=torch.long)
            store_idx = 0
            reservoir_samples = 0

        h_state = None # Initialize hidden state for the start of the sequence
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
                
                # 4. Compute controller output, passing and receiving the hidden state
                correction_logits, h_state = controller.forward(
                    noisy_state_A, noisy_state_B, 
                    setting_A_spin, setting_B_spin,
                    h_prev=h_state
                )

                # 5. Record the CHSH outcome
                if t >= washout_steps:
                    start_idx = (t - washout_steps) * batch_size
                    end_idx = start_idx + batch_size
                    
                    if readout_mode == 'end_to_end':
                        results_outputs_A[start_idx:end_idx] = torch.sign(correction_logits[:, 0])
                        results_outputs_B[start_idx:end_idx] = torch.sign(correction_logits[:, 1])
                    
                    results_settings[start_idx:end_idx] = all_settings_gpu[t].expand(batch_size, 2)
                    
                    # Store states for post-hoc training using reservoir sampling
                    if readout_mode == 'post_hoc':
                        # Each item in the batch is a potential sample
                        for i in range(batch_size):
                            reservoir_samples += 1
                            if store_idx < post_hoc_batch_size:
                                stored_states[store_idx] = torch.cat([state_A[i], state_B[i]], dim=-1)
                                stored_settings[store_idx] = all_settings_gpu[t]
                                store_idx += 1
                            else:
                                # If buffer is full, randomly replace an existing item
                                rand_idx = torch.randint(0, reservoir_samples, (1,)).item()
                                if rand_idx < post_hoc_batch_size:
                                    stored_states[rand_idx] = torch.cat([state_A[i], state_B[i]], dim=-1)
                                    stored_settings[rand_idx] = all_settings_gpu[t]

                # 6. Apply delay `d` for the substrate's actuation signal
                delayed_logits = delay_buffer.push(correction_logits)
                
                # 7. Evolve substrate
                substrate_dtype = next(substrate.parameters()).dtype
                substrate_input_A = self._build_input(setting_A_bit, delayed_logits[:, 0].unsqueeze(-1).to(substrate_dtype))
                substrate_input_B = self._build_input(setting_B_bit, delayed_logits[:, 1].unsqueeze(-1).to(substrate_dtype))
                substrate.step(substrate_input_A, substrate_input_B)

        # --- Post-Hoc Readout Training (if applicable) ---
        if readout_mode == 'post_hoc':
            # The stored data is already the correct batch size
            flat_states = stored_states
            flat_settings = stored_settings

            # Create a "teacher" signal: the ideal PR-Box outputs for the given settings.
            # Alice's output can be random, Bob's is derived from Alice's and the settings.
            # A simpler, powerful teacher is to train the readout to predict the settings themselves.
            target_A = (flat_settings[:, 0] * 2 - 1).float().unsqueeze(1) # Convert {0, 1} to {-1, 1}
            target_B = (flat_settings[:, 1] * 2 - 1).float().unsqueeze(1)
            targets = torch.cat([target_A, target_B], dim=1)

            # Analytically solve for the best possible linear readout using Ridge Regression.
            # This is equivalent to W_out = (X.T @ X + lambda*I)^-1 @ X.T @ Y
            # Using torch.linalg.lstsq is more stable and efficient.
            try:
                lambda_ridge = 1e-4
                # Solve (flat_states.T @ flat_states + lambda * I) W = flat_states.T @ targets
                # The solution to this is the least squares solution to a modified system.
                # PyTorch's lstsq solves AX = B for X.
                # A = flat_states, B = targets
                solution = torch.linalg.lstsq(flat_states, targets)
                W_out = solution.solution

                # Use this optimal readout to get the best possible outputs
                final_predicted_logits = flat_states @ W_out
                final_outputs_A = torch.sign(final_predicted_logits[:, 0])
                final_outputs_B = torch.sign(final_predicted_logits[:, 1])
                final_settings = flat_settings
                
                # Replace the main result tensors with the ones from the post-hoc evaluation
                results_outputs_A = final_outputs_A
                results_outputs_B = final_outputs_B
                results_settings = final_settings
            except torch.cuda.OutOfMemoryError:
                 logging.error("OOM error during post-hoc lstsq. The system is likely unstable. Returning worst fitness.")
                 return 4.0, {} # Return worst possible fitness
            except Exception as e:
                logging.error(f"Error during post-hoc lstsq: {e}. Returning worst fitness.")
                return 4.0, {} # Return worst possible fitness


        # 8. Scoring
        if readout_mode == 'post_hoc' and results_outputs_A is None:
            # Handle case where lstsq failed
            s_score = -4.0
            diagnostics = {"S_score": s_score, "error": "lstsq_failed"}
            return s_score, diagnostics
        if readout_mode == 'post_hoc':
            return self._compute_results(controller, substrate, results_outputs_A, results_outputs_B, flat_settings)
        else:
            return self._compute_results(controller, substrate, results_outputs_A, results_outputs_B, results_settings.view(-1, 2))

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