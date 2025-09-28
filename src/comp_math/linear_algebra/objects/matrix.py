import numpy as np
from typing import Union, List
from .vector import Vector

class Matrix:
    """Класс матрицы с операциями"""
    def __init__(self, data: Union[List[List], np.ndarray]):
        self._data = np.array(data, dtype=float)
        self.shape = self._data.shape
    
    def __getitem__(self, indices):
        return self._data[indices]
    
    def __setitem__(self, indices, value):
        self._data[indices] = value

    def __mul__(self, other):
        """Умножение: matrix * number или наоборот"""
        if isinstance(other, (int, float)):
            return self._multiply_scalar(other)
        else:
            return self.multiply(other)
    
    def __rmul__(self, other):
        """Умножение: number * matrix в обратном порядке"""
        if isinstance(other, (int, float)):
            return self._multiply_scalar(other)
        else:
            return NotImplemented
        
    def __truediv__(self, scalar):
        """Деление на скаляр: matrix / number"""
        if isinstance(scalar, (int, float)):
            return self._multiply_scalar(1.0 / scalar)
        else:
            return NotImplemented
    
    def add(self, other: 'Matrix') -> 'Matrix':
        """Сложение матриц через индексацию"""
        if self.shape != other.shape:
            raise ValueError("Размеры матриц не совпадают")
        
        result = Matrix(np.zeros(self.shape))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self[i, j] + other[i, j]
        return result
    
    def multiply(self, other: Union['Matrix', 'Vector', float]) -> Union['Matrix', 'Vector']:
        """Умножение матрицы на матрицу, вектор или скаляр"""
        if isinstance(other, (int, float)):
            return self._multiply_scalar(other)
        elif isinstance(other, Matrix):
            return self._multiply_matrix(other)
        elif isinstance(other, Vector):
            return self._multiply_vector(other)
        else:
            raise TypeError("Неподдерживаемый тип")
    
    def _multiply_scalar(self, scalar: float) -> 'Matrix':
        """Умножение на скаляр"""
        result = Matrix(np.zeros(self.shape))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self[i, j] * scalar
        return result
    
    def _multiply_matrix(self, other: 'Matrix') -> 'Matrix':
        """Умножение матрицы на матрицу"""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Несовместимые размеры матриц")
        
        result = Matrix(np.zeros((self.shape[0], other.shape[1])))
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                sum_val = 0.0
                for k in range(self.shape[1]):
                    sum_val += self[i, k] * other[k, j]
                result[i, j] = sum_val
        return result
    
    def _multiply_vector(self, vector: 'Vector') -> 'Vector':
        """Умножение матрицы на вектор"""
        if self.shape[1] != vector.dim:
            raise ValueError("Несовместимые размеры")
        
        result = Vector(np.zeros(self.shape[0]))
        for i in range(self.shape[0]):
            sum_val = 0.0
            for j in range(self.shape[1]):
                sum_val += self[i, j] * vector[j]
            result[i] = sum_val
        return result
    
    def inverse(self) -> 'Matrix':
        """Обращение матрицы методом Гаусса-Жордана"""
        A = self
        n = A.shape[0]
        
        # Инициализирунем расширенную матрицу
        augmented = np.zeros((n, 2*n))
        for i in range(n):
            for j in range(n):
                augmented[i, j] = A[i, j]
            augmented[i, n + i] = 1
        
        # Прямой ход
        for i in range(n):
            # Выбор главного элемента
            pivot_row = i
            for k in range(i + 1, n):
                if abs(augmented[k, i]) > abs(augmented[pivot_row, i]):
                    pivot_row = k
            
            # Перестановка строк
            if pivot_row != i:
                for j in range(2*n):
                    augmented[i, j], augmented[pivot_row, j] = augmented[pivot_row, j], augmented[i, j]
            
            # Нормировка текущей строки
            pivot = augmented[i, i]
            if pivot == 0:
                raise ValueError("Матрица вырождена")
            
            for j in range(2*n):
                augmented[i, j] /= pivot
            
            # Обнуление столбца
            for k in range(n):
                if k != i:
                    factor = augmented[k, i]
                    for j in range(2*n):
                        augmented[k, j] -= factor * augmented[i, j]
        
        # Извлекаем обратную матрицу
        A_inv = Matrix(np.zeros((n, n)))
        for i in range(n):
            for j in range(n):
                A_inv[i, j] = augmented[i, n + j]
        
        return A_inv
        
    def transpose(self) -> 'Matrix':
        """Транспонирование матрицы"""
        result = Matrix(np.zeros((self.shape[1], self.shape[0])))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[j, i] = self[i, j]
        return result
    
    def to_numpy(self) -> np.ndarray:
        """Конвертация в numpy array"""
        return self._data.copy()
