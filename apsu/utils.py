import numpy as np

def get_chsh_targets(settings_A, settings_B, seed=None):
    """
    Generates scientifically sound, stochastic binary targets for CHSH.

    In a real quantum experiment, each measurement is a random binary outcome.
    The expectation value (e.g., cos(theta)) emerges only after averaging
    many measurements. Providing the expectation value as a target for each
    shot is a form of information leak.

    This function simulates that process correctly. It calculates the ideal
    quantum probability of getting a '+1' outcome and then uses a seeded
    random number generator to produce a stream of +1/-1 binary targets
    that follow that probability distribution.

    Args:
        settings_A: Alice's measurement settings (0 or 1).
        settings_B: Bob's measurement settings (0 or 1).
        seed: A seed for the random number generator to ensure determinism.

    Returns:
        A tuple of (targets_A, targets_B) as numpy arrays of +1/-1.
    """
    rng = np.random.default_rng(seed)
    n_bits = len(settings_A)
    
    # Alice's target can be deterministically +1 without loss of generality.
    # The correlation depends on the product of outcomes.
    y_A = np.ones(n_bits)  
    y_B = np.zeros(n_bits)

    for i in range(n_bits):
        # Standard CHSH angles
        # Alice's angle choice: a=0, a'=pi/2
        # Bob's angle choice:   b=pi/4, b'=3pi/4
        angle_A = settings_A[i] * np.pi / 2.0
        angle_B = (np.pi / 4.0) if settings_B[i] == 0 else (3.0 * np.pi / 4.0)
        
        # The ideal QM correlation E[ab] = cos(a-b)
        correlation = np.cos(angle_A - angle_B)
        
        # The probability of Bob's outcome being the same as Alice's (+1)
        # is given by P(y_B=y_A) = (1 + E[ab]) / 2
        prob_same = (1 + correlation) / 2.0
        
        # Stochastically determine Bob's outcome based on this probability
        if rng.random() < prob_same:
            y_B[i] = y_A[i] # Same outcome as Alice
        else:
            y_B[i] = -y_A[i] # Different outcome
            
    return y_A, y_B 