import math
from typing import Callable, Tuple, Optional
import numpy as np

from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry
from ...base_nonlinear_solver import NonlinearSolver1D, NonlinearSolverND
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from ....linear_algebra.objects.matrix import Matrix
from ....linear_algebra.objects.vector import Vector
from comp_math.differentiation.numerical.numericalDifferentiator import NumericalDifferentiator
from comp_math.differentiation.numerical.numericalJacobian import NumericalJacobian


class NewtonSolver1D(NonlinearSolver1D):
    """Решение нелинейных уравнений методолм половинного деления"""
    
    def _solve_implementation_1d(self, f: Callable[[float], float],
                                interval: Tuple[float, float],
                                x_initial: float = None) -> float:
        a, b = interval
        
        if x_initial is None:
            x = (a + b) / 2
        else:
            # На случай если вышли за пределы локализации
            x = x_initial
            if x < a or x > b:
                x = (a + b) / 2

        for _ in range(self.max_iterations):
            fx = f(x)
            
            if abs(fx) < self.tolerance:
                return x
            
            dfx = NumericalDifferentiator.sixNodeDifferentiate(f, x, 0.1)
                
            x_new = x - fx / dfx
                
            error = abs(x_new - x)
            self._add_iteration(error)
            
            if error < self.tolerance:
                return x_new
                
            x = x_new
        
        return x
    
class NewtonSolverND(NonlinearSolverND):
    """Метод Ньютона для многомерных систем"""
    
    def _solve_implementation_nd(self, F: Callable[[np.ndarray], np.ndarray],
                               x0: np.ndarray,
                               J: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        x = x0.copy()
        n = len(x)
        
        for _ in range(self.max_iterations):
            Fx = F(x)
            
            if np.linalg.norm(Fx) < self.tolerance:
                return x
            
            if J is not None:
                Jx = J(x)
            else:
                Jx = NumericalJacobian.differentiate(F, x, 0.01)
            
            try:
                delta_x = np.linalg.solve(Jx, -Fx)
            except np.linalg.LinAlgError:
                delta_x = -np.linalg.pinv(Jx) @ Fx
            
            x_new = x + delta_x
            error = np.linalg.norm(delta_x)
            self._add_iteration(error)
            
            if error < self.tolerance:
                return x_new
                
            x = x_new
        
        return x
