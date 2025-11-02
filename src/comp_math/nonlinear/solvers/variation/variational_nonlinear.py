from typing import Callable, Tuple, List
import numpy as np

from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry
from comp_math.nonlinear.solvers.iterative.bisection_nonlinear import BisectionSolver
from ...base_nonlinear_solver import VariationalSolver1D


class VariationalEquationSolver1D(VariationalSolver1D):
    """Вариационный метод для решения нелинейных уравнений F(x) = 0"""
    
    def _solve_implementation_variational(self, 
                                        functional: Callable[[float], float],
                                        interval: Tuple[float, float],
                                        basis_functions: List[Callable],
                                        boundary_conditions: dict) -> float:
        """
        Решает F(x) = 0 через минимизацию функционала J[x] = ½F(x)²
        """
        a, b = interval
        n = len(basis_functions)
        
        A = np.zeros((n, n))
        b_vec = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                A[i, j] = self._integrate(
                    lambda x: basis_functions[i](x) * basis_functions[j](x), a, b
                )
        
        for i in range(n):
            b_vec[i] = self._integrate(
                lambda x: functional(x) * basis_functions[i](x), a, b
            )
        
        sbcg_solver = SLASolverRegistry.create_solver("sbcg")
        coefficients = sbcg_solver.solve(Matrix(A), Vector(b_vec))
        
        def approx_solution(x):
            return sum(c * phi(x) for c, phi in zip(coefficients, basis_functions))

        return self._find_root(functional, approx_solution, a, b)[0]
    
    def _find_root(self, F, approx_func, a, b, n_points=1000):
        """Находит корень уравнения F(x) = 0"""
        x_points = np.linspace(a, b, n_points)
        
        for i in range(len(x_points) - 1):
            x1, x2 = x_points[i], x_points[i+1]
            f1, f2 = F(x1), F(x2)
            
            if np.sign(f1) != np.sign(f2):
                solver = BisectionSolver()
                root = solver.solve(F, (x1, x2))
                return root
        
        min_idx = np.argmin([abs(F(x)) for x in x_points])
        return x_points[min_idx]
    
    def _integrate(self, f, a, b, n_points=1000):
        """Численное интегрирование"""
        x = np.linspace(a, b, n_points)
        y = [f(xi) for xi in x]
        return np.trapezoid(y, x)