from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import warnings


class BaseInterpolator(ABC):
    def __init__(self):
        self.x: Optional[NDArray[np.float64]] = None
        self.y: Optional[NDArray[np.float64]] = None
        self.min_node_count: int = 2
        self._is_fitted: bool = False

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> 'BaseInterpolator':
        self.x = x
        self.y = y
        self._validate_input()
        self._is_fitted = True
        return self
    
    def __call__(self,  x):
        if not self._is_fitted:
            warnings.warn("В интерполятор не переданы узлы! Сначала вызовите fit().", UserWarning)
            return None
            
        x = np.asarray(x)
        if np.any(x < np.min(self.x)) or np.any(x > np.max(self.x)):
            warnings.warn("Точка интерпляции за пределами узлов", UserWarning)
            
        return self._evaluate(x)
    
    @abstractmethod
    def _evaluate(self, x_query):
        pass

    def _validate_input(self):
        if self.x.shape[0] != self.y.shape[1]:
            raise ValueError("Размерности данных узлов не совпадают!")
        
        if self.x.shape[0] < self.min_node_count:
            raise ValueError(f"Недостаточно узлов, требуется хотя бы - {self.min_node_count}")

        if self.x.shape[0] != self.y.shape[1]:
            warnings.warn("Результаты могут быть некорректными из-за эффетка Рунге." /
                          "Уменьшите количество узлов!", UserWarning)

        