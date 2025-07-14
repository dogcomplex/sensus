import sys
import os
import json
import argparse
import logging

# Ensure the module is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apsu.classical_system_echotorch import ClassicalSystemEchoTorch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_phase0(config_path):
    """
    Executes Phase 0: Baseline Characterization.
    
    This phase runs the diagnostic checks on the ClassicalSystem to ensure
    the reservoirs are "healthy" before proceeding with optimization.
    
    Args:
        config_path (str): Path to the JSON configuration file.
    """
    logging.info("--- Starting Project Apsu: Phase 0 ---")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in configuration file: {config_path}")
        return

    system_config = config.get('classical_system_config', {})
    diagnostic_config = config.get('diagnostic_config', {})
    
    # Instantiate the system
    # For diagnostics, it's often better to run on CPU to avoid CUDA errors
    # if the GPU is busy, and since this is not a performance-critical path.
    device = diagnostic_config.get('device', 'cpu')
    system = ClassicalSystemEchoTorch(**system_config, device=device)
    logging.info(f"ClassicalSystem instantiated on device: {system.device}")
    
    # Run the diagnostic pre-flight check
    system.diagnose(
        steps=diagnostic_config.get('steps', 2000),
        plot_path=diagnostic_config.get('plot_path', 'apsu/diagnostics_report.png')
    )
    
    logging.info("--- Phase 0 Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 0 diagnostics for the Apsu project.")
    parser.add_argument(
        '--config',
        type=str,
        default='apsu/experiments/phase0/phase0_config.json',
        help='Path to the JSON configuration file for Phase 0.'
    )
    args = parser.parse_args()
    
    run_phase0(args.config)

if __name__ == "__main__":
    main()
