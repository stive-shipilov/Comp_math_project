import numpy as np
from ...base_SLA_solver import SLASolver
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from ....objects.matrix import Matrix
from ....objects.vector import Vector


class SBCGsolver(SLASolver):
    """Решение СЛАУ методом стабилизированных бисопряжённых
    градиентов"""
    
    def _solve_implementation(self, A: Matrix, b: Vector):
        n = b.dim
        x = Vector(np.zeros(n))
        r = b.subtract(A.multiply(x))
        rho, alpha, omega = 1, 1, 1
        r_ = r
        nu = Vector(np.zeros(n))
        p = nu

        # Итерации
        for iteration in range(1, self.max_iterations):
            rho_new = r_.scalar_mlp(r)
            beta = (rho_new*alpha)/(rho*omega)
            p = r.add(beta*(p.subtract(omega*nu)))
            nu = A.multiply(p)
            alpha = rho/(r_.scalar_mlp(nu))
            s = r.subtract(alpha*nu)
            t = A.multiply(s)
            omega = (t.scalar_mlp(s))/(t.scalar_mlp(t))
            x = x.add(omega*s).add(alpha*p)
            r = s.subtract(omega*t)  

            
            self._error = r.norm()
            self._iterations = iteration
            if self._error < self.tolerance:
                break

            rho = rho_new

        return x
