from abc import ABC, abstractmethod
import numpy as np
from comp_math.differentiation.baseDifferentiator import BaseDifferentiator
from comp_math.differentiation.numerical.numericalDifferentiator import NumericalDifferentiator
from typing import Callable
from numpy.typing import NDArray


class NumericalJacobian(BaseDifferentiator):
    """Класс для численного дифференцирования"""
        
    @staticmethod
    def differentiate(F: Callable[[NDArray], NDArray],
                        x: NDArray,
                        h: float = 1e-5) -> NDArray:
        """Численное вычисление якобиана"""
        n = len(x)
        J = np.zeros((n, n))
        F0 = F(x)
        
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = h
            J[:, j] = NumericalDifferentiator.sixNodeDifferentiate(F, x, h)
        
        return J
    