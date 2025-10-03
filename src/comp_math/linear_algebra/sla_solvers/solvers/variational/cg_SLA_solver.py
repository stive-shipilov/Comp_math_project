import numpy as np
from ...base_SLA_solver import SLASolver
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from ....objects.matrix import Matrix
from ....objects.vector import Vector


class CGsolver(SLASolver):
    """Решение СЛАУ методом сопряжённых градиентов"""
    
    def _solve_implementation(self, A: Matrix, b: Vector):
        n = b.dim
        x = Vector(np.zeros(n))
        r = b.subtract(A.multiply(x))
        z = b.subtract(A.multiply(x))

        # Итерации
        for iteration in range(1, self.max_iterations):
            alpha = r.scalar_mlp(r)/((A.multiply(z)).scalar_mlp(z))
            x = x.add(alpha*z)
            r_new = r.subtract(alpha*A.multiply(z))

            beta = (r_new.scalar_mlp(r_new))/(r.scalar_mlp(r))
            z = r_new.add(beta*z)

            
            self._error = r_new.norm()
            self._iterations = iteration
            if self._error < self.tolerance:
                break
            
            z = r_new.add(z*(beta))
            r = r_new

        return x
