import numpy as np
from ..base_interpolator import BaseInterpolator
from ...linear_algebra.objects.matrix import Matrix
from ...linear_algebra.objects.vector import Vector
from ...linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry


class CubicSpline(BaseInterpolator):
    def __init__(self):
        super().__init__()
        self.coeffs = None
        
    def _prepating_fitting(self, x, y):
        n = len(x)
        h = Vector([x[i+1] - x[i] for i in range(len(x)-1)])
        
        rhs = Vector(np.zeros(n))
        for i in range(1, n-1):
            rhs[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
        A = Matrix(np.zeros((n, n)))
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
        
        A[0, 0] = 1
        A[n-1, n-1] = 1
        
        solver = SLASolverRegistry.create_solver('gauss')
        M = solver.solve(A, rhs)
        
        # Вычисляем коэффициенты
        self.coeffs = []
        for i in range(n-1):
            a = y[i]
            b = (y[i+1] - y[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 6
            c = M[i] / 2
            d = (M[i+1] - M[i]) / (6 * h[i])
            self.coeffs.append([a, b, c, d])
        
        return self
    
    def _evaluate(self, x_query):
        self._prepating_fitting(self.x, self.y)
        x_query = Vector(np.asarray(x_query))
        result = Vector(np.zeros(x_query.dim))
        
        for i in range(0, x_query.dim):
            # Находим отрезок простым перебором
            idx = self._find_segment_for_point(x_query[i])
            a, b, c, d = self.coeffs[idx]
            dx = x_query[i] - self.x[idx]
            result[i] = a + b*dx + c*dx**2 + d*dx**3
        
        return result
    
    def _find_segment_for_point(self, x_val):
        """Находит индекс отрезка, содержащего точку x"""
        idx = 0
        for j in range(len(self.x) - 1):
            if self.x[j] <= x_val <= self.x[j + 1]:
                idx = j
                break
        # Если точка за пределами то будем брать крайний отрезко
        if x_val < self.x[0]:
            idx = 0
        elif x_val > self.x[-1]:
            idx = len(self.x) - 2
            
        return idx