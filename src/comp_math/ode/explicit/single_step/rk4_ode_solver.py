from typing import List, Tuple
from .base_runge_explicit_ode_solver import BaseRungeExplicitODESolver
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from ...base_ode_solver import BaseODESolver
import numpy as np


class Rk4ODESolver(BaseODESolver):
    """
    Решение методом RK4 (4й порядок)
    """
    
    def __init__(self):
        a = Matrix([[0, 0, 0, 0],
                    [1/2, 0, 0, 0],
                    [0, 1/2, 0, 0],
                    [0, 0, 1, 0]])
        b = Vector([1/6, 1/3, 1/3, 1/6])
        c = Vector([0, 1/2, 1/2, 1])

        self.solver = BaseRungeExplicitODESolver(a, b, c) 

    def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(f, t_span, y0, h)
