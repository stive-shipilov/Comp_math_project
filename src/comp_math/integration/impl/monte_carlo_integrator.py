from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray
from ..base_integrator import BaseIntegrator


class MonteCarloIntegrator(BaseIntegrator):
    """Метод Монте-Карло для интегрирования"""
    
    def __init__(self, n_points: int = 1000000):
        super().__init__(n_points)
        self.rng = np.random.default_rng()
        
    def integrate_func(self, func: Callable[[float], float], 
                      a: float, b: float, **kwargs) -> float:
        """Переопределенный метод для Монте-Карло"""
        self._validate_input(a, b)
        
        x_random = self.rng.uniform(a, b, self.n_points)
        y_random = func(x_random)
        
        # Оценка интеграла
        self.result = (b - a) * np.mean(y_random)
        
        # Оценка погрешности
        if self.n_points > 1:
            self.error_estimate = (b - a) * np.std(y_random) / np.sqrt(self.n_points)
        
        self._is_computed = True
        return self.result
    
    def _compute_integral(self, x: NDArray[np.float64], 
                         y: NDArray[np.float64],
                         a: float, b: float) -> float:
        """Монте-Карло по табличным данным - используем существующие точки как случайные"""
        self.result = (b - a) * np.mean(y)
        
        # Оценка погрешности
        if len(y) > 1:
            self.error_estimate = (b - a) * np.std(y) / np.sqrt(len(y))
        
        self._is_computed = True
        return self.result