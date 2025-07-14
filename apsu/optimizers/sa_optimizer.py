import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

class SAOptimizer(BaseOptimizer):
    """
    Simulated Annealing (SA) optimizer, using a custom implementation loop.
    This provides better real-time feedback than the scipy built-in.
    """

    def __init__(self, dimension, log_dir, n_iterations=100, initial_temp=1.0, cooling_rate=0.99, step_size=0.1, **kwargs):
        super().__init__(dimension, log_dir, **kwargs)
        self.n_iterations = n_iterations
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.history = []
        self.timestamps = []

    def run(self, fitness_function_wrapper, config):
        """
        Runs the full Simulated Annealing optimization loop.
        """
        logger.info(f"Starting Simulated Annealing optimization for {self.n_iterations} iterations.")
        
        # Initial solution
        if self.best_solution is None:
            self.best_solution = np.random.uniform(-0.1, 0.1, self.dimension)
            
        # Evaluate initial solution
        seed = int(time.time() * 1000) % (2**32 - 1)
        result = fitness_function_wrapper((self.best_solution, config, seed))
        self.best_fitness = result.get('fitness', -1.0)
        
        current_solution = self.best_solution
        current_fitness = self.best_fitness
        
        self.history.append(self.best_fitness)
        self.timestamps.append(time.time())

        for i in tqdm(range(self.n_iterations), desc="Simulated Annealing"):
            # Create a neighbor solution
            neighbor_solution = current_solution + np.random.randn(self.dimension) * self.step_size
            
            # Evaluate the neighbor
            seed = int(time.time() * 1000) % (2**32 - 1)
            result = fitness_function_wrapper((neighbor_solution, config, seed))
            neighbor_fitness = result.get('fitness', -1.0)
            
            # Acceptance probability
            if neighbor_fitness > current_fitness:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp((neighbor_fitness - current_fitness) / self.temp)
            
            # Decide whether to move to the neighbor
            if np.random.rand() < acceptance_prob:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
            
            # Update the best found solution
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.best_solution = current_solution
            
            # Cool down the temperature
            self.temp *= self.cooling_rate
            
            self.history.append(self.best_fitness)
            self.timestamps.append(time.time())

            if i % 10 == 0:
                logger.info(f"Iteration {i+1}/{self.n_iterations}: Best Fitness={self.best_fitness:.4f}, Current Temp={self.temp:.4f}")
                self.save_state()
                self.plot_progress(os.path.join(self.log_dir, "sa_progress.png"))

        logger.info("Simulated Annealing optimization finished.")
        self.save_state()
        self.plot_progress(os.path.join(self.log_dir, "sa_progress.png"))
        
        return self.best_solution, self.best_fitness

    def save_state(self):
        """Saves the optimizer history and best solution."""
        state = {
            'history': self.history,
            'timestamps': self.timestamps,
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness
        }
        with open(os.path.join(self.log_dir, 'optimizer_state.pkl'), 'wb') as f:
            pickle.dump(state, f)

    def plot_progress(self, save_path):
        """Plots the fitness history."""
        if not self.history:
            return
            
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.history, marker='.', linestyle='-', color='g')
        ax.set_title("Simulated Annealing Optimization Progress")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness (S-Score)")
        ax.grid(True, which='both', linestyle='--')
        plt.savefig(save_path)
        plt.close()
 