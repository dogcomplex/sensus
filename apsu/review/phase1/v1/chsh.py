import numpy as np

def get_chsh_settings(n_trials, seed=None):
    """
    Generates random measurement settings for a CHSH experiment.

    Alice's settings (a, a') are coded as {0, 1}.
    Bob's settings (b, b') are coded as {0, 1}.

    Args:
        n_trials (int): The number of measurement trials to generate.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        (np.ndarray, np.ndarray): Alice's settings, Bob's settings.
    """
    if seed is not None:
        np.random.seed(seed)
    
    alice_settings = np.random.randint(0, 2, size=n_trials)
    bob_settings = np.random.randint(0, 2, size=n_trials)
    
    return alice_settings, bob_settings

def get_chsh_targets(alice_settings, bob_settings):
    """
    Generates the 'correct' or 'target' outputs for a CHSH game.

    The CHSH inequality is maximized by correlations that follow the rule:
    a XOR b = x * y
    where a, b are the settings and x, y are the outputs {-1, 1}.
    We can simplify this to x = 1 and y = a XOR b.
    Or, for symmetry, x = y = a AND b.
    Let's use the XOR rule, which is standard. Alice's output can be
    random, and Bob's must be correlated. For simplicity, let's set
    Alice's target output x to be a simple function of her setting and
    Bob's target y to be a function of both settings.

    Let's define the target outputs x, y in {-1, 1} based on settings a, b in {0, 1}.
    A common quantum strategy gives the expectation value:
    <xy> = cos(theta_a - theta_b)
    With optimal angles, this leads to the outputs being correlated
    such that x*y = (-1)^(a*b).
    This means if a=1 and b=1, x and y should be anti-correlated, otherwise correlated.

    Let's set Alice's output x_target = 1 for all trials.
    Then Bob's output y_target must satisfy x*y = (-1)^(a*b), so y = (-1)^(a*b).

    Args:
        alice_settings (np.ndarray): Alice's settings {0, 1}.
        bob_settings (np.ndarray): Bob's settings {0, 1}.

    Returns:
        (np.ndarray, np.ndarray): Alice's target outputs, Bob's target outputs {-1, 1}.
    """
    n_trials = len(alice_settings)
    
    # Let Alice's target be random {-1, 1}
    # This is a valid strategy, as only the correlation matters.
    alice_targets = 2 * np.random.randint(0, 2, size=n_trials) - 1
    
    # Bob's target must be correlated with Alice's according to the CHSH game rule
    # a,b in {0,1}, x,y in {-1,1}
    # The rule for max violation is <xy> = (-1)^(ab)
    # So, y = x * (-1)^(a*b)
    bob_targets = alice_targets * ((-1)**(alice_settings * bob_settings))

    return alice_targets.reshape(-1, 1), bob_targets.reshape(-1, 1)

def calculate_s_score(outputs_A, outputs_B, settings_A, settings_B):
    """
    Calculates the CHSH S-score from experimental results.

    S = |E(0,0) + E(0,1) + E(1,0) - E(1,1)|

    where E(a,b) = <outputs_A * outputs_B> for settings (a,b).

    Args:
        outputs_A (np.ndarray): Alice's measured outputs {-1, 1}.
        outputs_B (np.ndarray): Bob's measured outputs {-1, 1}.
        settings_A (np.ndarray): Alice's settings {0, 1}.
        settings_B (np.ndarray): Bob's settings {0, 1}.

    Returns:
        float: The calculated S-score.
    """
    # Convert outputs from continuous values to {-1, 1}
    outputs_A = np.sign(outputs_A)
    outputs_B = np.sign(outputs_B)

    # Product of outcomes
    xy = outputs_A * outputs_B

    def expectation(a, b):
        indices = (settings_A == a) & (settings_B == b)
        if np.sum(indices) == 0:
            return 0.0 # No data for this setting combination
        return np.mean(xy[indices])

    # Calculate the four correlation terms
    E00 = expectation(0, 0)
    E01 = expectation(0, 1)
    E10 = expectation(1, 0)
    E11 = expectation(1, 1)

    # Calculate S score
    s = E00 + E01 + E10 - E11

    return abs(s) 