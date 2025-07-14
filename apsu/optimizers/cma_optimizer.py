import os
import time
import logging
from multiprocessing import Pool

import cma
import numpy as np
from tqdm import tqdm

from apsu.optimizers.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class CMAESOptimizer(BaseOptimizer):
    """
    An optimizer using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    """
    def __init__(self, dimension, log_dir, population_size=None, n_generations=100, sigma0=0.5, num_workers=None):
        super().__init__(dimension, log_dir)
        self.population_size = population_size if population_size is not None else 4 + int(3 * np.log(dimension))
        self.n_generations = n_generations
        self.sigma0 = sigma0
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.solutions = None # To hold the current population

        # CMA-ES specific initialization
        opts = cma.CMAOptions()
        opts.set('popsize', self.population_size)
        opts.set('CMA_diagonal', True)
        opts.set('seed', int(time.time()))
        # Set the log path for CMA's own log files
        # CMA logs to the current directory by default, so we give it a prefix
        # inside the designated log directory.
        log_path_prefix = os.path.join(str(log_dir), 'cma_')
        opts.set('verb_filenameprefix', log_path_prefix)

        self.es = cma.CMAEvolutionStrategy(dimension * [0], self.sigma0, opts)

        self.history = []

    def ask(self):
        """Asks the optimizer for a new population of candidate solutions."""
        self.solutions = self.es.ask()
        return self.solutions

    def tell(self, fitness_values):
        """Tells the optimizer the fitness values of the last population."""
        # CMA-ES minimizes, so we negate the fitness values which are S-scores
        self.es.tell(self.solutions, [-f for f in fitness_values])

        self.best_fitness = self.es.result.fbest * -1
        self.best_solution = self.es.result.xbest
        self.history.append(self.best_fitness)

    def run(self, fitness_function, config):
        """
        Runs the CMA-ES optimization loop by delegating to the BaseOptimizer.
        """
        super().run(fitness_function, config)
        logging.info("CMA-ES optimization finished.")
        # The base class doesn't return, but this one can for consistency if called directly.
        return self.best_solution, self.best_fitness
    
    def stop(self):
        return self.es.stop()

    def save_state(self):
        """Saves the optimizer state to a pickle file."""
        with open(os.path.join(self.log_dir, 'optimizer_state.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def plot_progress(self, save_path):
        """Plots the fitness history."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.history, marker='o', linestyle='-', color='b')
        ax.set_title("CMA-ES Optimization Progress")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness (S-Score)")
        ax.grid(True, which='both', linestyle='--')
        plt.savefig(save_path)
        plt.close() 