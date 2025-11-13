from abc import ABC, abstractmethod
from typing import Callable
from ..core.base_solver import BaseNumericalMethod


class BaseDifferentiator(BaseNumericalMethod):
    """Абстрактный класс для всех методов дифференцирования"""
    
    @abstractmethod
    def differentiate(self, f: Callable, x: float, **kwargs) -> float:
        pass