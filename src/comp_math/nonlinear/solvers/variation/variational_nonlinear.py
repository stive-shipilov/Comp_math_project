from typing import Callable, Tuple, List
import numpy as np

from comp_math.differentiation.numerical.numericalDifferentiator import NumericalDifferentiator
from comp_math.integration.integrator_registry import IntegratorRegistry
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry
from comp_math.nonlinear.solvers.iterative.bisection_nonlinear import BisectionSolver
from ...base_nonlinear_solver import VariationalSolver1D


class VariationalEquationSolver1D(VariationalSolver1D):
    """Решение F(x)=0 через минимизацию функционала J[u]"""
    
    def _solve_implementation_variational(self, F: Callable, interval: Tuple[float, float], 
                        basis_funcs: List[Callable]) -> float:
        """Решает F(x)=0 через минимизацию J = ∫[1/2 F^2 + 1/2λ(u')^2]dx"""
        a, b = interval
        n = len(basis_funcs)
        λ = 1e-3 
        
        # Матрица системы: A = M + λ*K
        # M_ij = ∫ F'(u_approx)·phi_i·phi_j dx
        # K_ij = ∫ phi_i'·phi_j' dx
        
        A = np.zeros((n, n))
        b_vec = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                def integrand_M(x):
                    return NumericalDifferentiator.sixNodeDifferentiate(F, x, 0.1) * basis_funcs[i](x) * basis_funcs[j](x)
                
                def integrand_K(x):
                    phi_i_prime = NumericalDifferentiator.sixNodeDifferentiate(basis_funcs[i], x, 0.1)
                    phi_j_prime = NumericalDifferentiator.sixNodeDifferentiate(basis_funcs[j], x, 0.1)
                    return phi_i_prime * phi_j_prime
                
                M_ij = self._integrate(integrand_M, a, b)
                K_ij = self._integrate(integrand_K, a, b)
                A[i,j] = M_ij + λ * K_ij
            
            # Правая часть: ∫ F(0)·phi_i dx
            b_vec[i] = -self._integrate(
                lambda x: F(0) * basis_funcs[i](x), a, b
            )
        
        solver = SLASolverRegistry.create_solver("sbcg")
        coeffs = solver.solve(Matrix(A), Vector(b_vec))
        
        # u(x) = Σ c_i·phi_i(x)
        # Корень F(x)=0 примерно точка минимума u(x)
        # Ищем минимум |u(x)| на [a,b]
        def u_approx(x):
            return sum(c * phi(x) for c, phi in zip(coeffs, basis_funcs))
        
        # Ищем где u(x) прмиерно равна 0 (т.е. минимум |u(x)|)
        x_test = np.linspace(a, b, 1000)
        u_vals = [abs(u_approx(x)) for x in x_test]
        root_idx = np.argmin(u_vals)
        
        # Доточняем методом Ньютона
        x0 = x_test[root_idx]
        for _ in range(10):
            fx = F(x0)
            fpx = NumericalDifferentiator.sixNodeDifferentiate(F, x0, 0.1)
            if abs(fpx) < 1e-12:
                break
            x0 = x0 - fx / fpx
        
        return x0
    
    def _integrate(self, f, a, b, n_points=1000):
        """Численное интегрирование"""
        x = np.linspace(a, b, n_points)
        y = [f(xi) for xi in x]
        integrator = IntegratorRegistry.create_solver("trapezoida")
        return integrator.integrate_table(y, x)