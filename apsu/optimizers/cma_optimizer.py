import cma
import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import time
from multiprocessing import Pool

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class CMAESOptimizer(BaseOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer.
    """

    def __init__(self, dimension, log_dir, population_size=None, n_generations=100, sigma0=0.5, num_workers=None):
        super().__init__(dimension)
        self.population_size = population_size if population_size is not None else 4 + int(3 * np.log(dimension))
        self.n_generations = n_generations
        self.sigma0 = sigma0
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()

        # CMA-ES specific initialization
        self.es = cma.CMAEvolutionStrategy(
            dimension * [0], 
            self.sigma0, 
            {
                'popsize': self.population_size,
                'CMA_diagonal': True,
                'seed': int(time.time())
            }
        )
        # The cma library expects a string for the log directory, not a Path object.
        log_dir_str = str(log_dir)
        self.logger = cma.CMADataLogger(log_dir_str).register(self.es)
        self.history = []

    def ask(self):
        """Asks the optimizer for a new population of candidate solutions."""
        return self.es.ask()

    def tell(self, fitness_values):
        """Tells the optimizer the fitness values of the last population."""
        # CMA-ES minimizes, so we negate the fitness values
        self.es.tell(self.solutions, [-f for f in fitness_values])
        self.logger.add()
        
        self.best_fitness = self.es.result.fbest * -1
        self.best_solution = self.es.result.xbest
        self.history.append(self.best_fitness)

    def run(self, fitness_function, config):
        """Runs the CMA-ES optimization loop."""
        with Pool(self.num_workers) as pool:
            for generation in range(self.n_generations):
                logging.info(f"--- Generation {generation + 1}/{self.n_generations} ---")
                solutions = self.es.ask()
                
                # Package arguments for the wrapper
                eval_args = [(sol, config) for sol in solutions]
                
                results = list(tqdm(pool.imap(fitness_function, eval_args), total=len(solutions), desc=f"Gen {generation+1}"))
                
                # Process results and tell the optimizer
                fitness_values = [res['fitness'] for res in results]
                self.es.tell(solutions, fitness_values)
                
                best_fitness_this_gen = self.es.result.fbest
                logging.info(f"Generation {generation + 1}: Best Fitness={best_fitness_this_gen:.4f}, Avg Fitness={np.mean(fitness_values):.4f}")
                self.logger.add()
                self.es.disp()

        logging.info("CMA-ES optimization finished.")
        return self.es.result.xbest, self.es.result.fbest
    
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