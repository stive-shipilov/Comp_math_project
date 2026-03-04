from typing import List, Tuple
from .base_runge_explicit_ode_solver import BaseRungeExplicitODESolver
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from ...base_ode_solver import BaseODESolver
import numpy as np


class EulerODESolver(BaseODESolver):
    """
    Решение методом Эйлера (1й порядок)
    """
    
    def __init__(self):
        a = Matrix([[0]])
        b = Vector([1])
        c = Vector([0])

        self.solver = BaseRungeExplicitODESolver(a, b, c) 

    def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(f, t_span, y0, h)
