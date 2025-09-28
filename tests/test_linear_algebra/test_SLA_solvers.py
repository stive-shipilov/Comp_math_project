import pytest
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry

class TestSLASolvers:
    """Тесты для решения СЛАУ"""
    
    def test_gauss_SLA_solver(self):
        """Тест создания вектора"""
        gauss_solver = SLASolverRegistry.create_solver("gauss")
        assert gauss_solver.solve(None, None) == 11

