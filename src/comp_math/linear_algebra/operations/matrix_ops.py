import numpy as np
from typing import Tuple
from numpy.typing import NDArray



class MatrixOperations:
    """Класс для операций с матрицами"""
    
    @staticmethod
    def LDUdecompose(A) -> Tuple[NDArray, NDArray, NDArray]:
        """LDU разложение матрицы"""
        U = MatrixOperations._extract_upper_triangular(A)
        L = MatrixOperations._extract_lower_triangular(A)
        D = MatrixOperations._extract_diag_matrix(A)

        return L, D, U
    
    @staticmethod
    def _extract_upper_triangular(A):
        n = len(A)
        U = np.zeros_like(A)
        
        for i in range(n):
            for j in range(i + 1, n):
                U[i, j] = A[i, j]
        
        return U
    
    @staticmethod
    def _extract_lower_triangular(A):
        n = len(A)
        L = np.zeros_like(A)
    
        for i in range(n):
            for j in range(0, i):
                L[i, j] = A[i, j]
        return L
    
    @staticmethod
    def _extract_diag_matrix(A):
        n = len(A)
        D = np.zeros_like(A)

        for i in range(n):
            D[i, i] = A[i, i]

        return D
    