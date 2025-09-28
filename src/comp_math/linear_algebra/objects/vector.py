import numpy as np
from typing import Union, List

class Vector:
    """Класс вектора с операциями через индексацию"""
    
    def __init__(self, data: Union[List, np.ndarray]):
        self._data = np.array(data, dtype=float).flatten()
        self.dim = len(self._data)
        
    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value
    
    def add(self, other: 'Vector') -> 'Vector':
        """Сложение векторов"""
        if self.dim != other.dim:
            raise ValueError("Размеры векторов не совпадают")
        
        result = Vector(np.zeros(self.dim))
        for i in range(self.dim):
            result[i] = self[i] + other[i]
        return result
    
    def scalar_mlp(self, other: 'Vector') -> float:
        """Скалярное произведение"""
        if self.dim != other.dim:
            raise ValueError("Размеры векторов не совпадают")
        
        result = 0.0
        for i in range(self.dim):
            result += self[i] * other[i]
        return result
    
    def multiply(self, scalar: float) -> 'Vector':
        """Умножение на скаляр"""
        result = Vector(np.zeros(self.dim))
        for i in range(self.dim):
            result[i] = self[i] * scalar
        return result
        
    def norm(self, p: int = 2) -> float:
        """Норма вектора"""
        if p == 2:  # Евклидова норма
            sum_sq = 0.0
            for i in range(self.dim):
                sum_sq += self[i] ** 2
            return np.sqrt(sum_sq)
        else:  # Lp норма
            sum_p = 0.0
            for i in range(self.dim):
                sum_p += abs(self[i]) ** p
            return sum_p ** (1/p)
        
    def to_numpy(self) -> np.ndarray:
        """Конвертация в numpy array"""
        return self._data.copy()