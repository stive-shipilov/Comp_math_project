import numpy as np
from ...base_SLA_solver import SLASolver
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from ....objects.matrix import Matrix
from ....objects.vector import Vector


class ZeidelSolver(SLASolver):
    """Решение СЛАУ методом Зейделя"""
    
    def _solve_implementation(self, A, b):
        n = b.dim
        x = Vector(np.zeros(n))
        
        L, D, U = MatrixOperations.LDUdecompose(A)


        B = -1*(L.add(D)).inverse().multiply(U)
        g = L.add(D).inverse().multiply(b)

        # Итерации
        for iteration in range(1, self.max_iterations + 1):
            x_new = B.multiply(x).add(g)
            
            self._add_error(x_new.subtract(x).norm())
            self._iterations = iteration
            x = x_new
            
            if self._last_error < self.tolerance:
                break
            
        return x
