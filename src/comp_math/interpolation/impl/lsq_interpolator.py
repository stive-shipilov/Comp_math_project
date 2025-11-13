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
        A_list = []
        for func in self.basis_functions:
            col = np.asarray(func(self.x))
            if col.ndim == 0:
                # если функция вернула скаляр, заполняем массивом !!
                col = np.full(self.x.shape, col)
            col = col.reshape(-1, 1)
            A_list.append(col)

        A = Matrix(np.hstack(A_list))

        AtA = A.transpose().multiply(A)
        AtY = A.transpose().multiply(Vector(self.y))
        
        # Решаем нормальные уравнения
        solver = SLASolverRegistry.create_solver('gauss')
        self.coefficients = solver.solve(AtA, AtY)
        self._is_fitted = True
        return self
        
    def _evaluate(self, x_query):
        self._get_coeffs()
        x_query = np.asarray(x_query)
        result = np.zeros_like(x_query, dtype=float)
        
        for coeff, func in zip(self.coefficients, self.basis_functions):
            result += coeff * func(x_query)

        return result