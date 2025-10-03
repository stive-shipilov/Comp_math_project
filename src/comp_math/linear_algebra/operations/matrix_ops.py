import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from ..objects.matrix import Matrix
from ..objects.vector import Vector


class MatrixOperations:
    """Класс для операций с матрицами"""
    
    @staticmethod
    def LDUdecompose(A: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
        """LDU разложение матрицы"""
        U = MatrixOperations._extract_upper_triangular(A)
        L = MatrixOperations._extract_lower_triangular(A)
        D = MatrixOperations._extract_diag_matrix(A)

        return L, D, U
    
    @staticmethod
    def get_spectral_radius(A: Matrix, max_iterations: int = 1000, 
                       tolerance=1e-10) -> float:
        """Cтепенной метод для нахождения спектрального радиуса матрицы
        - Используется задание центрированного случайного начального
        вектора, чтобы избежать плохое задание начального вектора 
        - Центрируем чтобы избежать того, что вектор
        будет ортогонален одному из собственных векторов
        - Используется отношение рэлея, которое позволяет более 
        точно оценить радиус
        """
        n = A.shape[0]
    
        # Начальный вектор
        v = Vector(np.random.rand(n) - 0.5)
        v = v / v.norm()
        
        lambda_prev = 0
        
        for i in range(max_iterations):
            w = A.multiply(v)
            w_norm = w.norm()
            
            # Если вектор равен 0 -> радиус тоде ноль
            if w_norm < 1e-12:
                return 0.0
            v_new = w / w_norm
            
            # Отношение Рэлея для более точной оценки радиуса
            lambda_new = v.scalar_mlp(w) / v.scalar_mlp(v)
            
            # Проверка сходимости алгоритма
            if abs(lambda_new - lambda_prev) < tolerance:
                break
                
            v = v_new
            lambda_prev = lambda_new
        
        return abs(lambda_new)

    
    @staticmethod
    def _extract_upper_triangular(A: Matrix):
        n = A.shape[0]
        U = Matrix(np.zeros_like(A.to_numpy()))
        
        for i in range(n):
            for j in range(i + 1, n):
                U[i, j] = A[i, j]
        
        return U
    
    @staticmethod
    def _extract_lower_triangular(A: Matrix):
        n = A.shape[0]
        L = Matrix(np.zeros_like(A.to_numpy()))
    
        for i in range(n):
            for j in range(0, i):
                L[i, j] = A[i, j]
        return L
    
    @staticmethod
    def _extract_diag_matrix(A: Matrix):
        n = A.shape[0]
        D = Matrix(np.zeros_like(A.to_numpy()))

        for i in range(n):
            D[i, i] = A[i, i]

        return D
        
