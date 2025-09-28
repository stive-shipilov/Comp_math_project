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

    def __mul__(self, other):
        """Умножение: vector * number или наоборот"""
        if isinstance(other, (int, float)):
            return self.multiply(other)
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        """Умножение: number * vector в обратном порядке"""
        if isinstance(other, (int, float)):
            return self.multiply(other)
        else:
            return NotImplemented
        
    def __truediv__(self, scalar):
        """Деление на скаляр: vector / number"""
        if isinstance(scalar, (int, float)):
            return self.multiply(1.0 / scalar)
        else:
            return NotImplemented
    
    def add(self, other: 'Vector') -> 'Vector':
        """Сложение векторов"""
        if self.dim != other.dim:
            raise ValueError("Размеры векторов не совпадают")
        
        result = Vector(np.zeros(self.dim))
        for i in range(self.dim):
            result[i] = self[i] + other[i]
        return result
    
    def subtract(self, other: 'Vector') -> 'Vector':
        """Вычитание векторов"""
        if self.dim != other.dim:
            raise ValueError("Размеры векторов не совпадают")
        
        result = Vector(np.zeros(self.dim))
        for i in range(self.dim):
            result[i] = self[i] - other[i]
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
        
    def norm(self, type: str = "decart") -> float:
        """Норма вектора"""
        match type:
            case "decart":
                sum_square = 0.0
                for i in range(self.dim):
                    sum_square += self[i] ** 2
                return sum_square**(0.5)
        
    def to_numpy(self) -> np.ndarray:
        """Конвертация в numpy array"""
        return self._data.copy()