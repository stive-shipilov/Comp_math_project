import numpy as np
from ...base_SLA_solver import SLASolver
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from comp_math.linear_algebra.operations.matrix_vector_ops import MatrixVectorOperations

class JacobiSolver(SLASolver):
    """Решение СЛАУ методом Якоби"""
    
    def _solve_implementation(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        x = np.zeros(n)
        
        L, D, U = MatrixOperations.LDUdecompose(A)

        D_inv = np.linalg.inv(D)

        B = -D_inv @ (L + U)
        g = MatrixVectorOperations.matvec(D_inv, b)   

        # Итерации
        for iteration in range(1, self.max_iterations + 1):
            x_new = MatrixVectorOperations.matvec(B, x) + g
            
            self._error = np.linalg.norm(x_new - x)
            self._iterations = iteration
            x = x_new
            
            if self._error < self.tolerance:
                break
            
        return x
