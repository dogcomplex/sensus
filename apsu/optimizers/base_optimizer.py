from abc import ABC, abstractmethod

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

    @abstractmethod
    def run(self, fitness_function, config):
        """
        The main entry point to start the optimization process.

        This method should contain the primary loop of the optimization algorithm.
        It will repeatedly generate candidate solutions, evaluate them using the
        provided fitness function, and update its internal state.

        Args:
            fitness_function (callable): The function to be maximized. It takes a
                candidate solution (e.g., a numpy array of weights) and the
                experiment config as input and returns a fitness score.
            config (dict): The full experiment configuration dictionary.
        
        Returns:
            tuple: A tuple containing the best found solution and its fitness score.
        """
        pass

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
