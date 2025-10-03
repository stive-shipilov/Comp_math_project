import numpy as np
from ...base_SLA_solver import SLASolver
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from ....objects.matrix import Matrix
from ....objects.vector import Vector


class BCGsolver(SLASolver):
    """Решение СЛАУ методом бисопряжённых градиентов"""
    
    def _solve_implementation(self, A: Matrix, b: Vector):
        n = b.dim
        x = Vector(np.zeros(n))
        r = b.subtract(A.multiply(x))
        z, p, s = r, r, r

        # Итерации
        for iteration in range(1, self.max_iterations):
            alpha = (p.scalar_mlp(r))/((A.multiply(z)).scalar_mlp(s))
            x = x.add(alpha*z)
            r_new = r.subtract(alpha*(A.multiply(z)))
            p_new = p.subtract(alpha*((A.transpose()).multiply(s)))
            beta = (p_new.scalar_mlp(r_new))/(p.scalar_mlp(r))
            z = r_new.add(beta*z)
            s = p_new.add(beta*s)

            
            self._error = r_new.norm()
            self._iterations = iteration
            if self._error < self.tolerance:
                break

            p = p_new
            r = r_new

        return x
