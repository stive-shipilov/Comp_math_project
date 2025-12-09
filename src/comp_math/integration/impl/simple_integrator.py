from typing import Callable
import numpy as np
from numpy.typing import NDArray
from ..base_integrator import BaseIntegrator


class RectangleIntegrator(BaseIntegrator):
    """Метод прямоугольников"""   
    def _compute_integral(self, x: NDArray[np.float64], 
                          y: NDArray[np.float64],
                          a: float, b: float) -> float:
        h = (b - a) / self.n_points        
        return h * np.sum(y)
    

class TrapezoidalIntegrator(BaseIntegrator):
    """Метод трапеций"""
    def _compute_integral(self, x: NDArray[np.float64], 
                          y: NDArray[np.float64],
                          a: float, b: float) -> float:        
        h = (b - a) / (self.n_points - 1)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    

class SimpsonIntegrator(BaseIntegrator):
    """Метод Симпсона"""
    def __init__(self, n_points: int = 101):
        # Для метода Симпсона надо нечетное количество точек
        if n_points % 2 == 0:
            n_points += 1
            print(f"Метод Симпсона требует нечетного числа точек. Добавлена еще одна точка. Теперь их n={n_points}")
        super().__init__(n_points)
        
    def _compute_integral(self, x: NDArray[np.float64], 
                          y: NDArray[np.float64],
                          a: float, b: float) -> float:        
        n = len(x)
        
        segments = n - 1
        h = (b - a) / segments
        
        coeffs = np.ones(n)
        coeffs[1:-1:2] = 4
        coeffs[2:-2:2] = 2
        
        return (h / 3) * np.sum(coeffs * y)
