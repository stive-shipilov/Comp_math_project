from typing import Callable
import numpy as np
from ..base_integrator import BaseIntegrator


class RectangleIntegrator(BaseIntegrator):
    """Метод прямоугольников"""   
    def _compute_integral(self, func: Callable[[float], float], 
                          a: float, b: float) -> float:
        h = (b - a) / self.n_points
        
        x = np.linspace(a + h/2, b - h/2, self.n_points)
        
        y = func(x)
        return h * np.sum(y)
    

class TrapezoidalIntegrator(BaseIntegrator):
    """Метод трапеций"""
    def _compute_integral(self, func: Callable[[float], float], 
                          a: float, b: float, **kwargs) -> float:
        x = np.linspace(a, b, self.n_points)
        y = func(x)
        
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
        
    def _compute_integral(self, func: Callable[[float], float], 
                          a: float, b: float, **kwargs) -> float:
        x = np.linspace(a, b, self.n_points)
        y = func(x)
        
        n = self.n_points - 1  # количество отрезков должно быть четным
        h = (b - a) / n
        
        # Коэффициенты: 1 для крайних, 4 для нечетных, 2 для четных
        coeffs = np.ones(self.n_points)
        coeffs[1:-1:2] = 4
        coeffs[2:-2:2] = 2
        
        return (h / 3) * np.sum(coeffs * y)
