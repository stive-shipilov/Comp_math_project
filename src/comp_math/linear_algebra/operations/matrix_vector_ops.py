import numpy as np
from numpy.typing import NDArray


class MatrixVectorOperations:
    """Класс для операций между матрицами и векторами"""
    
    @staticmethod
    def matvec(A: NDArray, 
               x: NDArray) -> NDArray:
        """Умножение матрицы на вектор: A*x"""
        A_rows, A_cols = A.shape
        x_dim = x.shape[0]
        result_vec = np.zeros(A_rows)
        print(result_vec)
        for i in range(0, A_rows):
            element = 0
            for j in range(0, x_dim):
                element += A[i][j]*x[j]
            result_vec[i] = element


        return result_vec
    
