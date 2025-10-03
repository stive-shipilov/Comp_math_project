import math
import numpy as np
from ...base_SLA_solver import SLASolver
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from ....objects.matrix import Matrix
from ....objects.vector import Vector


class RelaxationSolver(SLASolver):
    """Решение СЛАУ методом релаксации"""
    
    def _solve_implementation(self, A, b):
        n = b.dim
        x = Vector(np.zeros(n))
        
        L, D, U = MatrixOperations.LDUdecompose(A)

        # Вычисляем релаксационный параметр
        T = -1*D.inverse().multiply(L.add(U))
        spectrum_radius = MatrixOperations.get_spectral_radius(T)
        omega = 1.5  # значение по умолчанию

        # В зависимости от спектрального радиуса, выбираем разный параметр
        if spectrum_radius < 1:
            omega = 2 / (1 + math.sqrt(1 - spectrum_radius**2))
            
        omega = 2 / (1 + math.sqrt(1 - spectrum_radius**2))

        B = (D.add(omega*L)).inverse().multiply(((1-omega)*D).subtract(omega*U))
        g = omega*(D.add(omega*L)).inverse().multiply(b)

        # Итерации
        for iteration in range(1, self.max_iterations + 1):
            x_new = B.multiply(x).add(g)
            
            self._add_error(x_new.subtract(x).norm())
            self._iterations = iteration
            x = x_new
            
            if self._last_error < self.tolerance:
                break
            
        return x
