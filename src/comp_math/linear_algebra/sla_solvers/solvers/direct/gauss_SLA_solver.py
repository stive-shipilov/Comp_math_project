import numpy as np
from ...base_SLA_solver import SLASolver


class GaussSolver(SLASolver):
    """Решение СЛАУ методом Гаусса"""
    
    def _solve_implementation(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        x = np.zeros(n)

        # Прямой ход
        for i in range(0, n):
            for j in range(i+1, n):
                k = A[j][i]/A[i][i]
                A[j] = A[j] - k*A[i]
                b[j] = b[j] - k*b[i]
        # Обратный ход
        for i in range(n-1, -1, -1):
            x[i] = (b[i] - np.sum(A[i][(i+1):]*x[(i+1):]))/A[i][i]

        return x
