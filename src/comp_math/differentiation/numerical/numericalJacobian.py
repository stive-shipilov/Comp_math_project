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
        """
        Численное вычисление якобиана методом центральных разностей
        """
        n = len(x)
        F0 = F(x)
        m = len(F0)
        J = np.zeros((m, n))
        
        for j in range(n):
            # Создаем возмущенный вектор
            x_plus = x.copy()
            x_plus[j] += h
            F_plus = F(x_plus)
            
            x_minus = x.copy()
            x_minus[j] -= h
            F_minus = F(x_minus)
            
            # Центральная разность
            J[:, j] = (F_plus - F_minus) / (2 * h)
            
            # Проверка на слишком малые значения
            if np.all(np.abs(J[:, j]) < 1e-12):
                # Если центральная разность дала ноль, пробуем одностороннюю
                J[:, j] = (F_plus - F0) / h
        
        return J
    