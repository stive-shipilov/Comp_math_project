import numpy as np
from typing import Tuple
from numpy.typing import NDArray



class MatrixOperations:
    """Класс для операций с матрицами"""
    
    @staticmethod
    def LDUdecompose(self, A) -> Tuple[NDArray, NDArray, NDArray]:
        """LDU разложение матрицы"""
        U = self._extract_upper_triangular(A)
        L = self._extract_lower_triangular(A)
        D = self._extract_diag_matrix(A)

        return U
    
    def _extract_upper_triangular(A):
        n = len(A)
        U = np.zeros_like(A)
        
        for i in range(n):
            for j in range(i, n):
                U[i, j] = A[i, j]
        
        return U
    
    def _extract_lower_triangular(A):
        n = len(A)
        L = np.zeros_like(A)
    
        for i in range(n):
            for j in range(0, i + 1):
                L[i, j] = A[i, j]
        return L
    
    def _extract_diag_matrix(A):
        n = len(A)
        D = np.zeros_like(A)

        for i in range(n):
            D[i, i] = A[i, i]
    
        
        
