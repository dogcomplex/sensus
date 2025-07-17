import numpy as np

def calculate_chsh_score(
    outputs_A: list[int], 
    outputs_B: list[int], 
    settings: list[tuple[int, int]],
    bootstrap_seed: int | None = None,
    n_boot: int = 1000
) -> tuple[float, dict[tuple[int, int], float]]:
    """
    Calculates the S-score from lists of outcomes and settings.

    Args:
        outputs_A: List of {-1, +1} outcomes for party A.
        outputs_B: List of {-1, +1} outcomes for party B.
        settings: List of (a, b) tuples where a,b are in {0, 1}.

    Returns:
        A tuple containing:
        - The calculated S-score.
        - A dictionary of the four correlation values E(a,b).
    """
    yA = np.asarray(outputs_A)
    yB = np.asarray(outputs_B)
    s = np.asarray(settings)
    
    if not (len(yA) == len(yB) == len(s)):
        raise ValueError("All input lists must have the same length.")

    def get_s_from_indices(indices):
        yA_s = yA[indices]
        yB_s = yB[indices]
        s_s = s[indices]
        correlations = {}
        for a_val in (0, 1):
            for b_val in (0, 1):
                mask = (s_s[:, 0] == a_val) & (s_s[:, 1] == b_val)
                if not np.any(mask):
                    correlations[(a_val, b_val)] = 0.0
                else:
                    correlations[(a_val, b_val)] = np.mean(yA_s[mask] * yB_s[mask])

        s_score = (correlations.get((0, 0), 0) + 
                   correlations.get((0, 1), 0) + 
                   correlations.get((1, 0), 0) - 
                   correlations.get((1, 1), 0))
        return abs(s_score), correlations

    # Calculate S-score for the original data
    s_score_observed, correlations_observed = get_s_from_indices(np.arange(len(yA)))

    if bootstrap_seed is not None:
        rng = np.random.default_rng(bootstrap_seed)
        bootstrap_scores = [get_s_from_indices(rng.choice(len(yA), len(yA), replace=True))[0] for _ in range(n_boot)]
        # For now, just return the observed score, but one could return CIs as well
        # In the context of the optimizer, we want the point estimate.
        # The CI is more for final analysis.
    
    return s_score_observed, correlations_observed


def calculate_nonsignaling_metric(
    outputs_local: list[int], 
    settings_local: list[int], 
    settings_remote: list[int],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> tuple[float, tuple[float, float]]:
    """
    Calculates the non-signaling deviation metric (Delta_NS) for one party.
    It measures the maximum change in the local party's output probability
    when the remote party's setting changes.

    Args:
        outputs_local: {-1, +1} outcomes for the local party.
        settings_local: {0, 1} settings for the local party.
        settings_remote: {0, 1} settings for the remote party.
        n_boot: Number of bootstrap resamples for confidence interval.
        ci: Confidence interval level.
        seed: Seed for the bootstrap RNG.

    Returns:
        A tuple containing:
        - The calculated Delta_NS metric.
        - A tuple with the (lower, upper) bounds of the confidence interval.
    """
    out = np.asarray(outputs_local)
    s_loc = np.asarray(settings_local)
    s_rem = np.asarray(settings_remote)

    # Convert {-1,+1} to {0,1} for probability calculation
    out_prob = (out + 1) / 2 

    def get_delta(indices):
        max_delta = 0.0
        # Iterate over local settings
        for a_val in (0, 1):
            # P(y=+1 | a, b=0)
            mask0 = (s_loc[indices] == a_val) & (s_rem[indices] == 0)
            p0 = np.mean(out_prob[indices][mask0]) if np.any(mask0) else 0.5
            
            # P(y=+1 | a, b=1)
            mask1 = (s_loc[indices] == a_val) & (s_rem[indices] == 1)
            p1 = np.mean(out_prob[indices][mask1]) if np.any(mask1) else 0.5
            
            max_delta = max(max_delta, abs(p0 - p1))
        return max_delta

    # Calculate metric for the original data
    delta_ns_observed = get_delta(np.arange(len(out)))

    # Bootstrap for confidence interval
    rng = np.random.default_rng(seed)
    bootstrap_deltas = [get_delta(rng.choice(len(out), len(out), replace=True)) for _ in range(n_boot)]
    
    lower_bound = np.percentile(bootstrap_deltas, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_deltas, (1 + ci) / 2 * 100)

    return delta_ns_observed, (lower_bound, upper_bound) 