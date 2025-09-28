import numpy as np
import pytest
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry

class TestSLASolvers:
    """Тесты для решения СЛАУ"""
    
    def test_gauss_SLA_solver(self):
        """Тест создания вектора"""
        gauss_solver = SLASolverRegistry.create_solver("gauss")

        # Тестовая СЛАУ
        x_true = np.array([1.0, 2.0, 3.0])
        
        A = np.array([
            [4.0, 1.0, 1.0],
            [1.0, 5.0, 2.0], 
            [1.0, 2.0, 6.0]
        ])
        
        b = A @ x_true
        x_calc = gauss_solver.solve(A, b)
        assert np.allclose(x_calc, x_true, atol=1e-10)
