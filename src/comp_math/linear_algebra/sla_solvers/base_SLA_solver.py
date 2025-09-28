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
    
    def solve(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Публичный метод решения с единой точкой валидации"""
        A_np, b_np = self._validate_and_convert_input(A, b)
        self._prepare_solver(A_np, b_np)
        return self._solve_implementation(A_np, b_np)
    
    @abstractmethod
    def _solve_implementation(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Метод джля реализации алгоритма"""
        pass
    
    def _validate_and_convert_input(self, A, b) -> Tuple[np.ndarray, np.ndarray]:
        """Единая валидация и преобразование входных данных"""
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float)
        
        # Проверка размеров
        if A_np.ndim != 2:
            raise ValueError("Матрица A должна быть двумерной")
        if b_np.ndim != 1:
            raise ValueError("Вектор b должен быть одномерным")
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("Матрица A должна быть квадратной")
        if A_np.shape[0] != b_np.shape[0]:
            raise ValueError("Размеры A и b не совпадают")
        if A_np.shape[0] == 0:
            raise ValueError("Система не может быть пустой")
        
        return A_np, b_np
    
    def _prepare_solver(self, A: np.ndarray, b: np.ndarray):
        """Подготовка решателя"""
        self._iterations = 0
        self._error = 0.0

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