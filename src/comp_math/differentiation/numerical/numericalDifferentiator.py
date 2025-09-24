from abc import ABC, abstractmethod
from comp_math.differentiation.baseDifferentiator import BaseDifferentiator
from typing import Callable


class NumericalDifferentiator(BaseDifferentiator):
    """Класс для численного дифференцирования"""

    @staticmethod
    def leftDifferentiate(f: Callable, x: float, h: float = None) -> float:
        """Дифференцирование по формуле (f(x+h)-f(x))/h"""
        return (f(x+h)-f(x))/h

    @staticmethod
    def rightDifferentiate(f: Callable, x: float, h: float = None) -> float:
        """Дифференцирование по формуле ((f(x)-f(x-h))/h"""
        return (f(x)-f(x-h))/h

    @staticmethod
    def doubleSideDifferentiate(f: Callable, x: float, h: float = None) -> float:
        """Дифференцирование по формуле (f(x+h)-f(x-h))/2*h"""
        return (f(x+h)-f(x-h))/(2*h)

    @staticmethod
    def fourNodeDifferentiate(f: Callable, x: float, h: float = None) -> float:
        """Дифференцирование по формуле 
        4/3 * (f(x+h) - f(x-h)) / (2h) - 1/3 * (f(x+2h) - f(x-2h)) / (4h)
        """
        return (4/3)*(f(x+h)-f(x-h))/(2*h) - (1/3)*(f(x+2*h)-f(x-2*h))/(4*h)

    @staticmethod
    def sixNodeDifferentiate(f: Callable, x: float, h: float = None) -> float:
      """Дифференцирование по формуле 
        (3/2) * (f(x+h) - f(x-h)) / (2h) - (3/5) * (f(x+2h) - f(x-2h)) /
        (4h) + (1/10) * (f(x+3h) - f(x-3h)) / (6h)"""
      result = (3/2)*(f(x+h)-f(x-h))/(2*h) - (3/5)*(f(x+2*h)-f(x-2*h))/(4*h) + (1/10)*(f(x+3*h)-f(x-3*h))/(6*h)
      return result
