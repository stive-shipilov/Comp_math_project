import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from ..objects.matrix import Matrix


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
        
