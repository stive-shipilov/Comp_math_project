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
        
    def integrate_func(self, func: Callable[[float], float], 
                      a: float, b: float, **kwargs) -> float:
        """Переопределенный метод для Гаусса"""
        self._validate_input(a, b)
        x_transformed = 0.5 * (b - a) * self.nodes + 0.5 * (a + b)
        y = func(x_transformed)
        weighted_sum = np.dot(self.weights, y)
        self.result = 0.5 * (b - a) * weighted_sum
        self._is_computed = True
        return self.result
    
    def integrate_table(self, x: NDArray[np.float64], 
                        y: NDArray[np.float64]) -> float:
        """Гаусс не поддерживает табличные данные напрямую"""
        raise NotImplementedError(
            "Метод Гаусса не поддерживает интегрирование по табличным данным напрямую. "
            "Нужно для этого использовать интерполяции."
        )
    
    def _compute_integral(self, x: NDArray[np.float64], 
                         y: NDArray[np.float64],
                         a: float, b: float) -> float:
        """Для совместимости с базовым классом"""
        raise NotImplementedError(
            "Метод Гаусса требует использования integrate_func с функцией"
        )