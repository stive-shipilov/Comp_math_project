import numpy as np
import pytest
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations

class TestVectorOperations:
    """Тесты для операций с векторами"""
    
    def test_matrix_vector_multiplication(self):
        """Тест создания вектора"""
        # Проверка умножения матрицы на вектор
        A = Matrix([[1, 3],
                      [4, 5]])
        x = Vector([1, 2])
        assert A.multiply(x).to_numpy().tolist() == [7, 14]

        # Проверка разложения LDU
        A = Matrix([[1, 3, 4],
                    [4, 5, 6],
                    [1, 2, 3]])
        L, D, U = MatrixOperations.LDUdecompose(A)
        
        expected_L = np.array([[0, 0, 0],
                            [4, 0, 0],
                            [1, 2, 0]])
        
        expected_D = np.array([[1, 0, 0],
                            [0, 5, 0],
                            [0, 0, 3]])
        
        expected_U = np.array([[0, 3, 4],
                            [0, 0, 6],
                            [0, 0, 0]])
        
        # Используем np.array_equal для точного сравнения
        assert np.array_equal(L.to_numpy(), expected_L)
        assert np.array_equal(D.to_numpy(), expected_D)
        assert np.array_equal(U.to_numpy(), expected_U) 
    
    def test_matrix_inversion(self):
        A = Matrix([[4, 1, 1],
                    [1, 5, 2],
                    [1, 2, 6]])

        A_inv = A.inverse()

        result = A.multiply(A_inv)
        expected = np.eye(3)
        assert np.allclose(result.to_numpy(), expected, atol = 1e-15)
