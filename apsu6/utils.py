import numpy as np
from pathlib import Path
import logging

def bit_to_spin(bit: int) -> int:
    """Map {0,1} -> {-1,+1}; raises on invalid."""
    if bit == 1:
        return 1
    if bit == 0:
        return -1
    raise ValueError(f"Input bit must be 0 or 1, but got {bit}")

def bits_to_spins(bits: tuple[int, int]) -> tuple[int, int]:
    """
    Accepts iterable/tuple of length 2 of {0,1}.
    Returns (spin_A, spin_B) in {-1,+1}.
    """
    if len(bits) != 2:
        raise ValueError(f"Input must be a tuple of length 2, but got {len(bits)}")
    a, b = bits
    return bit_to_spin(a), bit_to_spin(b)

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