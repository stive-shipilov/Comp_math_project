from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

class BaseODESolver(ABC):
    """Абстрактный базовый класс для всех решателей ОДУ."""
    
    @abstractmethod
    def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        """Абстрактный метод решения ОДУ"""
        pass