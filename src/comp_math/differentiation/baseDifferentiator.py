from abc import ABC, abstractmethod
from typing import Callable

class BaseDifferentiator(ABC):
    """Абстрактный класс для всех методов дифференцирования"""
    
    @abstractmethod
    def differentiate(self, f: Callable, x: float, **kwargs) -> float:
        pass