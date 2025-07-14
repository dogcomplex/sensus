from abc import ABC, abstractmethod
import os
import logging
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import json

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    
    This class defines the common interface that the experiment harness will use
    to run an optimization experiment. Each specific optimizer (e.g., CMA-ES,
    Simulated Annealing) must inherit from this class and implement the
    `run` method.
    """

    def __init__(self, dimension, log_dir, **kwargs):
        """
        Initializes the base optimizer.

        Args:
            dimension (int): The number of parameters to optimize.
            log_dir (str): The directory where logs and results should be saved.
            **kwargs: Additional optimizer-specific parameters.
        """
        self.dimension = dimension
        self.log_dir = log_dir
        self.best_fitness = -float('inf')
        self.best_solution = None
        self.n_generations = 0
        self.num_workers = 1

    @abstractmethod
    def ask(self):
        """Asks the optimizer for a new population of candidate solutions."""
        pass

    @abstractmethod
    def tell(self, fitness_values):
        """Tells the optimizer the fitness values of the last population."""
        pass
    
    @abstractmethod
    def stop(self):
        """Checks if the optimizer's stopping criteria are met."""
        return False

    def run(self, fitness_function, run_path, payload_id):
        """
        The main entry point to start the optimization process.
        
        The 'config' or 'eval_payload' is no longer passed directly. Instead,
        we pass the path to the run directory and a unique ID for the definitive
        payload file, which the fitness_function wrapper will read from disk.
        """
        # We still need to configure the optimizer itself, so we load the payload here
        # just for that purpose.
        payload_path = run_path / f"payload_{payload_id}.json"
        with open(payload_path, 'r') as f:
            eval_payload = json.load(f)
        
        optimizer_config = eval_payload.get('optimizer', {}).get('config', {})
        self.n_generations = optimizer_config.get('n_generations', 100)
        self.num_workers = optimizer_config.get('num_workers', os.cpu_count())

        with Pool(processes=self.num_workers) as pool:
            for generation in range(self.n_generations):
                if self.stop():
                    logging.info("Stopping criteria met. Terminating optimization.")
                    break
                
                logging.info(f"--- Generation {generation + 1}/{self.n_generations} ---")
                
                solutions = self.ask()
                # Pass the run path and payload ID to the wrapper.
                eval_args = [(sol, str(run_path), payload_id) for sol in solutions]
                
                results = list(tqdm(pool.imap(fitness_function, eval_args), total=len(solutions), desc=f"Gen {generation+1}"))
                
                fitness_values = [res['fitness'] for res in results]
                self.tell(fitness_values)

                logging.info(f"Generation {generation + 1}: Best Fitness={self.best_fitness:.4f}, Avg Fitness={np.mean(fitness_values):.4f}")

    def save_state(self):
        """
        Saves the current state of the optimizer to disk.
        This is crucial for resuming long-running experiments.
        Default implementation does nothing, should be overridden.
        """
        pass

    def load_state(self):
        """
        Loads the optimizer state from disk.
        Default implementation does nothing, should be overridden.
        """
        pass

    def plot_progress(self, save_path):
        """
        Generates a plot of the optimization progress.
        Default implementation does nothing, should be overridden.
        """
        pass
