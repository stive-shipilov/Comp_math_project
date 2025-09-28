import numpy as np
import pytest
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector

class TestSLASolvers:
    """Тесты для решения СЛАУ"""
    
    def test_gauss_SLA_solver(self):
        """Тест создания вектора"""
        gauss_solver = SLASolverRegistry.create_solver("gauss")
        jacobi_solver = SLASolverRegistry.create_solver("jacobi")

        # Тестовая СЛАУ
        x_true = Vector([1.0, 2.0, 3.0])
        
        A = Matrix([
            [4.0, 1.0, 1.0],
            [1.0, 5.0, 2.0], 
            [1.0, 2.0, 6.0]
        ])
        
        b = A.multiply(x_true)
        assert np.allclose(gauss_solver.solve(A, b).to_numpy(), x_true.to_numpy(), atol=1e-10)
        assert np.allclose(jacobi_solver.solve(A, b).to_numpy(), x_true.to_numpy(), atol=1e-10)

