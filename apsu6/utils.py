import numpy as np
from pathlib import Path
import logging
import torch

def bit_to_spin(bit: int) -> int:
    """Map {0,1} -> {-1,+1}; raises on invalid."""
    if bit == 1:
        return 1
    if bit == 0:
        return -1
    raise ValueError(f"Input bit must be 0 or 1, but got {bit}")

def bits_to_spins(bits: torch.Tensor) -> torch.Tensor:
    """
    Vectorized conversion of a tensor of bits {0, 1} to spins {-1, +1}.
    Args:
        bits (torch.Tensor): A tensor of any shape containing 0s and 1s.
    Returns:
        A tensor of the same shape with 0s mapped to -1 and 1s to +1.
    """
    return 2 * bits.float() - 1

def load_chsh_settings(filepath: str | Path) -> np.ndarray:
    """
    Loads a binary file containing CHSH settings.
    Each trial is 2 bytes (uint8, uint8).
    """
    p = Path(filepath)
    if not p.exists():
        logging.error(f"Randomness file not found at: {p}")
        raise FileNotFoundError(f"Could not find CHSH settings file: {p}")
    
    with open(p, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Reshape into pairs of settings (a, b)
    return data.reshape(-1, 2) 