import numpy as np
import torch
import logging

def calculate_chsh_score(
    outputs_A: torch.Tensor, 
    outputs_B: torch.Tensor, 
    settings: torch.Tensor,
    bootstrap_seed: int | None = None, # Keep for potential future use
    n_boot: int = 1000
) -> tuple[torch.Tensor, dict[tuple[int, int], torch.Tensor]]:
    """
    Calculates the S-score from tensors of outcomes and settings on the GPU.
    """
    if not (outputs_A.shape[0] == outputs_B.shape[0] == settings.shape[0]):
        raise ValueError("All input tensors must have the same length.")

    correlations = {}
    for a_val in (0, 1):
        for b_val in (0, 1):
            mask = (settings[:, 0] == a_val) & (settings[:, 1] == b_val)
            if not torch.any(mask):
                correlations[(a_val, b_val)] = torch.tensor(0.0, device=outputs_A.device)
            else:
                # Use .double() for high-precision calculation of the mean
                correlations[(a_val, b_val)] = torch.mean((outputs_A[mask] * outputs_B[mask]).double())

    s_score = (correlations.get((0, 0), 0).double() + 
               correlations.get((0, 1), 0).double() + 
               correlations.get((1, 0), 0).double() - 
               correlations.get((1, 1), 0).double())
    
    return torch.abs(s_score), correlations


def calculate_s_score_with_bootstrap(
    outputs_A: torch.Tensor,
    outputs_B: torch.Tensor,
    settings: torch.Tensor,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    use_cpu_fallback: bool = False
) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """
    Calculates the S-score with a memory-efficient, parallelized bootstrap on the GPU.
    """
    if use_cpu_fallback:
        # Transfer to CPU and perform a slower, sequential bootstrap
        return _calculate_s_score_bootstrap_cpu(outputs_A, outputs_B, settings, n_boot, ci, seed)

    device = outputs_A.device
    num_trials = len(outputs_A)
    
    # --- Observed Score ---
    s_score_observed, correlations_observed = calculate_chsh_score(outputs_A, outputs_B, settings)

    # --- Bootstrap CI and p-values ---
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate all bootstrap indices at once
    bootstrap_indices = torch.randint(0, num_trials, (n_boot, num_trials), generator=generator, device=device)
    
    # Gather the data for all bootstrap samples. This is memory-intensive, but
    # should be manageable if the intermediate masks are handled carefully.
    s_A_boot = outputs_A[bootstrap_indices]
    s_B_boot = outputs_B[bootstrap_indices]
    settings_boot = settings[bootstrap_indices]
    
    # Calculate products once
    products = s_A_boot * s_B_boot
    
    bootstrap_s_scores = torch.zeros(n_boot, device=device, dtype=torch.double)
    
    # Process each CHSH setting pair in a vectorized way to manage memory
    setting_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a_val, b_val in setting_pairs:
        # Create mask for the current setting pair. This is the main memory bottleneck.
        # By doing it inside the loop, the mask is released after each iteration.
        mask = (settings_boot[..., 0] == a_val) & (settings_boot[..., 1] == b_val)
        
        # Calculate correlations for all bootstrap samples in parallel for this setting
        safe_mask_sum = mask.sum(dim=1, dtype=torch.double).clamp(min=1)
        masked_products_sum = (products * mask).sum(dim=1, dtype=torch.double)
        correlations = masked_products_sum / safe_mask_sum

        # Apply CHSH formula
        if a_val == 1 and b_val == 1:
            bootstrap_s_scores -= correlations
        else:
            bootstrap_s_scores += correlations

    bootstrap_s_scores = torch.abs(bootstrap_s_scores)
    
    # P-value calculation and confidence intervals (as before)
    p_classical = (bootstrap_s_scores > 2.0).double().mean()
    p_tsirelson = (bootstrap_s_scores > 2.828427).double().mean()
    q = torch.tensor([(1 - ci) / 2, (1 + ci) / 2], device=device, dtype=torch.double)
    ci_bounds = torch.quantile(bootstrap_s_scores, q)
    
    return s_score_observed, correlations_observed, ci_bounds, p_classical, p_tsirelson

def _calculate_s_score_bootstrap_cpu(outputs_A, outputs_B, settings, n_boot, ci, seed):
    """Internal CPU-based sequential bootstrap for fallback."""
    device = torch.device('cpu')
    outputs_A = outputs_A.to(device)
    outputs_B = outputs_B.to(device)
    settings = settings.to(device)
    logging.info("Using CPU fallback for bootstrap calculation.")

    num_trials = len(outputs_A)
    s_score_observed, correlations_observed = calculate_chsh_score(outputs_A, outputs_B, settings)

    generator = torch.Generator(device=device).manual_seed(seed)
    bootstrap_s_scores = torch.zeros(n_boot, device=device, dtype=torch.double)

    for i in range(n_boot):
        indices = torch.randint(0, num_trials, (num_trials,), generator=generator, device=device)
        s_A_sample = outputs_A[indices]
        s_B_sample = outputs_B[indices]
        settings_sample = settings[indices]
        score, _ = calculate_chsh_score(s_A_sample, s_B_sample, settings_sample)
        bootstrap_s_scores[i] = score

    p_classical = (bootstrap_s_scores > 2.0).double().mean()
    p_tsirelson = (bootstrap_s_scores > 2.828427).double().mean()
    q = torch.tensor([(1 - ci) / 2, (1 + ci) / 2], device=device, dtype=torch.double)
    ci_bounds = torch.quantile(bootstrap_s_scores, q)
    
    return s_score_observed.to(outputs_A.device), correlations_observed, ci_bounds.to(outputs_A.device), p_classical.to(outputs_A.device), p_tsirelson.to(outputs_A.device)


def calculate_teacher_loss_gpu(
    outputs_A: torch.Tensor, 
    outputs_B: torch.Tensor, 
    settings: torch.Tensor
) -> torch.Tensor:
    """Calculates the PR-Box teacher loss entirely on the GPU."""
    o_A_bits = (outputs_A.float() + 1) / 2
    o_B_bits = (outputs_B.float() + 1) / 2
    s_A_bits = settings[:, 0]
    s_B_bits = settings[:, 1]
    
    actual_xor = torch.bitwise_xor(o_A_bits.long(), o_B_bits.long())
    target_and = torch.bitwise_and(s_A_bits.long(), s_B_bits.long())
    
    return torch.mean((actual_xor - target_and).float()**2)


def calculate_nonsignaling_metric_gpu(
    outputs_local: torch.Tensor, 
    settings_local: torch.Tensor, 
    settings_remote: torch.Tensor,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized, GPU-based calculation of the non-signaling metric and its CI.
    """
    device = outputs_local.device
    out_prob = (outputs_local.float() + 1) / 2

    def get_delta_vectorized(indices_batch):
        # indices_batch shape: (n_boot, n_trials)
        n_boot_local, n_trials = indices_batch.shape
        
        # Gather the data for all bootstrap samples at once
        s_loc_boot = settings_local[indices_batch] # (n_boot, n_trials)
        s_rem_boot = settings_remote[indices_batch] # (n_boot, n_trials)
        out_prob_boot = out_prob[indices_batch]     # (n_boot, n_trials)

        max_deltas = torch.zeros(n_boot_local, device=device)
        for a_val in (0, 1):
            mask0 = (s_loc_boot == a_val) & (s_rem_boot == 0)
            mask1 = (s_loc_boot == a_val) & (s_rem_boot == 1)
            
            # Calculate means for all bootstrap samples in parallel
            p0 = (out_prob_boot * mask0).sum(dim=1) / mask0.sum(dim=1).clamp(min=1)
            p1 = (out_prob_boot * mask1).sum(dim=1) / mask1.sum(dim=1).clamp(min=1)
            
            deltas = torch.abs(p0 - p1)
            max_deltas = torch.maximum(max_deltas, deltas)
            
        return max_deltas

    # Calculate metric for the original data
    observed_indices = torch.arange(len(out_prob), device=device).unsqueeze(0)
    delta_ns_observed = get_delta_vectorized(observed_indices)[0]

    # Bootstrap for confidence interval
    generator = torch.Generator(device=device).manual_seed(seed)
    bootstrap_indices = torch.randint(0, len(out_prob), (n_boot, len(out_prob)), generator=generator, device=device)
    bootstrap_deltas = get_delta_vectorized(bootstrap_indices)
    
    # Calculate CI bounds using torch.quantile
    q = torch.tensor([(1 - ci) / 2, (1 + ci) / 2], device=device)
    ci_bounds = torch.quantile(bootstrap_deltas, q)
    
    return delta_ns_observed, ci_bounds 