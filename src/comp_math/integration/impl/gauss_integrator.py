from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss
from ..base_integrator import BaseIntegrator


class GaussIntegrator(BaseIntegrator):
    """Квадратурный метод Гаусса-Лежандра"""
    
    def __init__(self, n_points: int = 5):
        super().__init__(n_points)
        self.nodes, self.weights = self._get_gauss_legendre_params(n_points)
        
    def _get_gauss_legendre_params(self, n: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Получение узлов и весов для квадратур Гаусса-Лежандра"""
        # Используем numpy для получения узлов и весов
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(n)
        return nodes, weights
        
    def _compute_integral(self, func: Callable[[float], float], 
                      a: float, b: float, **kwargs) -> float:
        # Преобразуем каждый узел
        x_transformed = []
        for node in self.nodes:
            transformed = 0.5 * (b - a) * node + 0.5 * (a + b)
            x_transformed.append(transformed)
        
        y = func(np.array(x_transformed))
        
        weighted_sum = 0.0
        for i in range(len(self.weights)):
            weighted_sum += self.weights[i] * y[i]
        
        return 0.5 * (b - a) * weighted_sum