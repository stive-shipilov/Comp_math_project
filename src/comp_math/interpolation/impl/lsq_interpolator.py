import numpy as np
from ..base_interpolator import BaseInterpolator
from ...linear_algebra.objects.matrix import Matrix
from ...linear_algebra.objects.vector import Vector
from ...linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry


class UniversalLSQ(BaseInterpolator):
    def __init__(self, basis_functions=None):
        super().__init__()
        self.basis_functions = None
        if basis_functions is None:
            # По умолчанию полином степени 2
            self.basis_functions = [
                lambda x: 1,
                lambda x: x,
                lambda x: x**2
            ]
        else:
            self.basis_functions = basis_functions
            
        self.coefficients = None
        self._is_fitted = False
        
    def _get_coeffs(self):
        A = Matrix(np.zeros((len(self.x), len(self.basis_functions))))
        col = Vector(np.zeros(len(self.x)))
        for k, func in enumerate(self.basis_functions):
            for i in range(0, len(self.x)):
                col[i] = func(self.x[i])
            if col.dim == 1:
                # если функция вернула скаляр, заполняем массивом !!
                for i in range(0, len(self.x)-1):
                    col[i] = func(self.x[i])
            for i in range(0, col.dim):
                A[i, k] = col[i]

        AtA = A.transpose().multiply(A)
        AtY = A.transpose().multiply(Vector(self.y))
        
        # Решаем нормальные уравнения
        solver = SLASolverRegistry.create_solver('gauss')
        self.coefficients = solver.solve(AtA, AtY)
        self._is_fitted = True
        return self
        
    def _evaluate(self, x_query):
        self._get_coeffs()
        x_query = Vector(np.asarray(x_query))
        result = Vector(np.zeros(x_query.dim))
        
        for coeff, func in zip(self.coefficients, self.basis_functions):
            for i in range(0, x_query.dim):
                result[i] += coeff * func(x_query[i])

        return result