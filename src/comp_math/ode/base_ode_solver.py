from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

class BaseODESolver(ABC):
    """Абстрактный базовый класс для всех решателей ОДУ."""
    
    def __init__(self, f, dim):
        self.f = f
        self.dim = dim
    
    @abstractmethod
    def solve(self, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        """Абстрактный метод решения ОДУ"""
        pass