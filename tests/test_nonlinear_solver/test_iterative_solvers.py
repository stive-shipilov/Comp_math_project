import numpy as np
import pytest
from comp_math.nonlinear.nonlinear_solvers_registry import NonlinearSolverRegistry
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector

class TestSLASolvers:
    """Тесты для решения нелинйеных уравнений"""
    
    def test_1D_solvers(self):
        """Тест создания вектора"""
        bisection_solver = NonlinearSolverRegistry.create_solver("bisection")
        newton1D_solver = NonlinearSolverRegistry.create_solver("newton1D")
        fixed_points1D_solver = NonlinearSolverRegistry.create_solver("fixedPoints1D", lambda_param=0.01)

        """f(x) = x^2 - 4, корни: -2, 2"""
        def quadratic(x): return x**2 - 4

        """f(x) = x³ - 6x² + 11x - 6, корни: 1, 2, 3"""
        def polynomial_3_roots(x): return x**3 - 6*x**2 + 11*x -  6
        
        roots = bisection_solver.solve(quadratic, (-1e3, 1e3), grid_points = 1000, x_initial = [1, 1])
        expected_roots = [-2, 2]
        assert np.allclose(roots, expected_roots, atol=1e-10)

        roots = bisection_solver.solve(polynomial_3_roots, (-1e3, 1e3), grid_points = 100000)
        expected_roots = [1, 2, 3]
        assert np.allclose(roots, expected_roots, atol=1e-10)

        roots = newton1D_solver.solve(quadratic, (-1e3, 1e3), grid_points = 1000, x_initial = [1, 1])
        expected_roots = [-2, 2]
        assert np.allclose(roots, expected_roots, atol=1e-10)

        roots = newton1D_solver.solve(polynomial_3_roots, (-1e3, 1e3), grid_points = 100000)
        expected_roots = [1, 2, 3]
        assert np.allclose(roots, expected_roots, atol=1e-10)

        roots = fixed_points1D_solver.solve(quadratic, (-1e3, 1e3), grid_points = 100000)
        expected_roots = [-2, 2]
        print(roots)
        assert np.allclose(roots, expected_roots, atol=1e-10)

        roots = fixed_points1D_solver.solve(polynomial_3_roots, (-1e3, 1e3), grid_points = 1000000)
        expected_roots = [1, 2, 3]
        assert np.allclose(roots, expected_roots, atol=1e-10)


    def test_multidimensional_newton(self):
        """Тест многомерного метода Ньютона"""
        newtonND_solver = NonlinearSolverRegistry.create_solver("newtonND")

        # Тест 1: Система 2x2 - пересечение окружности и прямой
        # x^2 + y^2 = 1
        # x - y = 0
        # Решения: (sqrt(2)/2, sqrt(2)/2) и (-sqrt(2)/2, -sqrt(2)/2)
        def system1(x):
            return np.array([
                x[0]**2 + x[1]**2 - 1,
                x[0] - x[1]
            ])

        solution1 = newtonND_solver.solve(system1, np.array([1.0, 1.0]))
        expected1 = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
        assert np.allclose(solution1, expected1, atol=1e-6)
        assert np.allclose(system1(solution1), np.array([0, 0]), atol=1e-9)

        solution2 = newtonND_solver.solve(system1, np.array([-0.5, -0.5]))
        expected2 = np.array([-np.sqrt(2)/2, -np.sqrt(2)/2])
        assert np.allclose(solution2, expected2, atol=1e-6)
        assert np.allclose(system1(solution2), np.array([0, 0]), atol=1e-9)
