import math
from typing import Callable, Tuple
import numpy as np
from ...base_nonlinear_solver import NonlinearSolver1D
from comp_math.linear_algebra.operations.matrix_ops import MatrixOperations
from ....linear_algebra.objects.matrix import Matrix
from ....linear_algebra.objects.vector import Vector


class BisectionSolver(NonlinearSolver1D):
    """Решение нелинейных уравнений методолм половинного деления"""
    
    def _solve_implementation_1d(self, f: Callable[[float], float],
                                interval: Tuple[float, float],
                                x_initial: float = 0) -> float:
        a, b = interval
        fa, fb = f(a), f(b)
        
        # Если случайно так получилось, что корень попал на край интервала локализайии
        if abs(fa) < self.tolerance: return a
        if abs(fb) < self.tolerance: return b
        
        for _ in range(self.max_iterations):
            c = (a + b) / 2
            fc = f(c)
            error = (b - a) / 2
            
            self._add_iteration(error)
            
            if abs(fc) < self.tolerance or error < self.tolerance:
                return c
            
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        
        return (a + b) / 2