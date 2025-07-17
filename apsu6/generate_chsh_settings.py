import numpy as np
from pathlib import Path
import argparse

def generate_chsh_settings(num_trials: int, output_file: Path):
    """
    Generates a balanced, shuffled dataset of CHSH measurement settings.

    The CHSH game requires a dataset where each of the four possible input
    setting combinations ((0,0), (0,1), (1,0), (1,1)) is presented an
    equal number of times. This function creates such a dataset and shuffles
    it randomly to prevent any temporal ordering artifacts in the experiment.

    Args:
        num_trials (int): The total number of trials (must be a multiple of 4).
        output_file (Path): The path to save the binary output file.
    """
    if num_trials % 4 != 0:
        raise ValueError("Total number of trials must be a multiple of 4 for a balanced CHSH experiment.")

    # Calculate the number of times each setting pair should appear.
    trials_per_setting = num_trials // 4

    # Create the base list of setting pairs.
    settings = []
    settings.extend([(0, 0)] * trials_per_setting)
    settings.extend([(0, 1)] * trials_per_setting)
    settings.extend([(1, 0)] * trials_per_setting)
    settings.extend([(1, 1)] * trials_per_setting)

    # Convert to a NumPy array for shuffling and saving.
    settings_array = np.array(settings, dtype=np.uint8)

    # Shuffle the settings randomly. This is crucial.
    # We use a dedicated RNG for reproducibility.
    rng = np.random.default_rng(seed=42)
    rng.shuffle(settings_array)

    # Ensure the output directory exists.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the settings to a binary file.
    with open(output_file, 'wb') as f:
        f.write(settings_array.tobytes())

    print(f"Successfully generated {num_trials} balanced CHSH settings.")
    print(f"Saved to: {output_file}")
    # Verification
    counts = np.unique(settings_array, axis=0, return_counts=True)
    print("Verification counts:")
    for setting, count in zip(counts[0], counts[1]):
        print(f"  Setting {tuple(setting)}: {count} times")


def main():
    parser = argparse.ArgumentParser(description="Generate a balanced CHSH settings file for Project Apsu.")
    parser.add_argument(
        '--trials',
        type=int,
        default=4000,
        help="Total number of trials to generate. Must be a multiple of 4."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='apsu6/data/chsh_settings.bin',
        help="Path to the output binary file."
    )
    args = parser.parse_args()

    generate_chsh_settings(num_trials=args.trials, output_file=Path(args.output))

if __name__ == "__main__":
    main() 