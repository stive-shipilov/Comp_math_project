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
        "Cтепенной метод для нахождения спектрального радиуса матрицы"
        n = A.shape[0]
        m = 0
        v = Vector(np.ones(n))
        w = Vector(np.zeros(n))
        for i in range(max_iterations):
            w = A.multiply(v)
            m_new = w.norm()
            error =  m_new - m
            m = m_new
            v = w/m
            if np.abs(error) < tolerance:
                break

        return m

    
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
        
