import numpy as np
import pytest
from comp_math.nonlinear.nonlinear_solvers_registry import NonlinearSolverRegistry


def test_variational_equation_solver():
    
    solver = NonlinearSolverRegistry.create_solver("variational")

    """f(x) = x^2 - 4, корни: -2, 2"""
    def quadratic(x): return x**2 - 4
    
    # Базисные функции
    basis = [lambda x: 1]
    
    roots = solver.solve(
        functional=quadratic,
        search_area=(-3, 3),
        grid_points=1000,
        basis_functions=basis
    )
    
    
    found_roots = sorted(roots)
    expected_roots = [-2, 2]
    
    assert np.allclose(roots, expected_roots, atol=1e-4)
    
