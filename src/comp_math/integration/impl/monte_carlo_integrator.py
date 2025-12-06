from typing import Callable, Optional
import numpy as np
from ..base_integrator import BaseIntegrator


class MonteCarloIntegrator(BaseIntegrator):
    """Метод Монте-Карло для интегрирования"""
    
    def __init__(self, n_points: int = 1000000):
        super().__init__(n_points)
        self.rng = np.random.default_rng()
        
    def _compute_integral(self, func: Callable[[float], float], 
                          a: float, b: float, **kwargs) -> float:
        x_random = self.rng.uniform(a, b, self.n_points)
        y_random = func(x_random)
        
        # Оценка интеграла
        integral_estimate = (b - a) * np.mean(y_random)
        
        # Оценка погрешностии
        if self.n_points > 1:
            self.error_estimate = (b - a) * np.std(y_random) / np.sqrt(self.n_points)
        
        return integral_estimate