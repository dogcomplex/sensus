import cma
import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class CMAESOptimizer(BaseOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer.
    """

    def __init__(self, dimension, log_dir, sigma0=0.5, population_size=None, n_generations=100, **kwargs):
        super().__init__(dimension, log_dir, **kwargs)
        self.sigma0 = sigma0
        self.population_size = population_size
        self.n_generations = n_generations
        
        # CMA-ES specific initialization
        self.es = cma.CMAEvolutionStrategy(
            dimension * [0], 
            self.sigma0, 
            {
                'popsize': self.population_size,
                'CMA_diagonal': True  # Use diagonal covariance matrix to save memory
            }
        )
        self.logger = cma.CMADataLogger(os.path.join(self.log_dir, "cma_")).register(self.es)
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

    def run(self, fitness_function_wrapper, config):
        """
        Runs the full CMA-ES optimization loop.
        """
        logger.info(f"Starting CMA-ES optimization for {self.n_generations} generations.")
        
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            
        num_workers = config.get("num_workers", os.cpu_count())
        pool = multiprocessing.Pool(processes=num_workers)

        try:
            for generation in range(self.n_generations):
                logger.info(f"--- Generation {generation+1}/{self.n_generations} ---")
                
                self.solutions = self.ask()
                
                eval_args = [(sol, config, (generation * len(self.solutions)) + i) for i, sol in enumerate(self.solutions)]
                
                results = list(tqdm(pool.imap(fitness_function_wrapper, eval_args), total=len(self.solutions), desc=f"Gen {generation+1}"))
                
                fitness_values = [res['fitness'] for res in results]
                self.tell(fitness_values)

                avg_fitness_in_gen = np.mean(fitness_values) if fitness_values else -1
                logger.info(f"Generation {generation+1}: Best Fitness={self.best_fitness:.4f}, Avg Fitness={avg_fitness_in_gen:.4f}")

                self.save_state()
                self.plot_progress(os.path.join(self.log_dir, "cma_progress.png"))

        finally:
            pool.close()
            pool.join()
        
        logger.info("CMA-ES optimization finished.")
        return self.best_solution, self.best_fitness

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