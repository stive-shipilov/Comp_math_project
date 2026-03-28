import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import root


class BVProblem(ABC):
    """Абстрактный базовый класс для краевой задачи"""
    
    def __init__(self, domain: Tuple[float, float]):
        self.domain = domain
        self.a, self.b = domain
    
    @abstractmethod
    def get_type(self) -> str:
        """Возвращает тип задачи"""
        pass
    
    def get_domain(self) -> Tuple[float, float]:
        return self.domain
