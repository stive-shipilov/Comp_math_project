from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray
import warnings

from comp_math.linear_algebra.objects.vector import Vector
from ..core.base_solver import BaseNumericalMethod


class BaseIntegrator(BaseNumericalMethod):
    """Базовй класс для реализации методов численного интегрирования"""
    def __init__(self,  n_points: int = 10000):
        self.n_points = n_points
        self.result: Optional[float] = None
        self.error_estimate: Optional[float] = None
        self._is_computed: bool = False

    def integrate(self, func: Callable[[float], float], 
                  a: float, b: float, **kwargs) -> float:
        """Вычисляет интеграл функции func от a до b"""
        self._validate_input(a, b)
        self.result = self._compute_integral(func, a, b, **kwargs)
        self._is_computed = True
        return self.result
    
    def _validate_input(self, a: float, b: float):
        """Проверка корректности входных данных"""
        if a >= b:
            raise ValueError(f"Неверные пределы интегрирования: a={a} >= b={b}")
    
    @abstractmethod
    def _compute_integral(self, func: Callable[[float], float], 
                          a: float, b: float, **kwargs) -> float:
        """Основная логика вычисления интеграла (реализуется в подклассах)"""
        pass
        
        