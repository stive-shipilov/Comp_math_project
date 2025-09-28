from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class SLASolver(ABC):
    """Абстрактный базовый класс для решателей СЛАУ"""
    
    def __init__(self, max_iterations=1000, tolerance=1e-10):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._iterations = 0
        self._error = 0.0
    
    @abstractmethod
    def solve(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Решает систему уравнений Ax = b"""
        pass
    
    @property
    def iterations_count(self) -> int:
        """Количество выполненных итераций"""
        return self._iterations
    
    @property
    def last_error(self) -> float:
        """Последняя достигнутая погрешность"""
        return self._error
    
    def validate_input(self, A, b):
        """Проверка корректности входных данных"""
        if len(A) != len(b):
            raise ValueError("Размеры матрицы A и вектора b не совпадают")
        if any(len(row) != len(A) for row in A):
            raise ValueError("Матрица A должна быть квадратной")