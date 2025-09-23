import numpy as np
import pytest
from comp_math.linear_algebra.operations.matrix_vector_ops import MatrixVectorOperations


class TestVectorOperations:
    """Тесты для операций с векторами"""
    
    def test_matrix_vector_multiplication(self):
        """Тест создания вектора"""
        A = np.array([[1, 3],
                      [4, 5]])
        x = np.array([1, 2])
        assert MatrixVectorOperations.matvec(A, x).tolist() == [7, 14]
