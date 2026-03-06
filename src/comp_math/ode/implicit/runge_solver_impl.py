from typing import List, Tuple
from .base_runge_implicit_ode_solver import BaseRungeImplicitODESolver
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from ..base_ode_solver import BaseODESolver
import numpy as np


class GaussLegendre2ODESolver(BaseODESolver):
    """
    Решение методом Гаусса-Лужандра (2й порядок)
    """
    
    def __init__(self):
        sqrt3_6 = np.sqrt(3)/6
        a = Matrix([[1/4, 1/4 - sqrt3_6],
                    [1/4 + sqrt3_6, 1/4]])
        b = Vector([1/2, 1/2])
        c = Vector([1/2 - sqrt3_6, 1/2 + sqrt3_6])

        self.solver = BaseRungeImplicitODESolver(a, b, c) 

    def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(f, t_span, y0, h)


class GaussLegendre4ODESolver(BaseODESolver):
    """
    Решение методом Гаусса-Лежандра (4й порядок)
    """
    
    def __init__(self):
        sqrt15 = np.sqrt(15)
        a = Matrix([[5/36, 2/9 - sqrt15/15, 5/36 - sqrt15/30],
                    [5/36 + sqrt15/24, 2/9, 5/36 - sqrt15/24],
                    [5/36 + sqrt15/30, 2/9 + sqrt15/15, 5/36]])
        b = Vector([5/18, 4/9, 5/18])
        c = Vector([1/2 - sqrt15/10, 1/2, 1/2 + sqrt15/10])

        self.solver = BaseRungeImplicitODESolver(a, b, c) 

    def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(f, t_span, y0, h)


class RadoIIAODESolver(BaseODESolver):
    """
    Решение методом Радо IIA (3й порядок, L-устойчив)
    """
    
    def __init__(self):
        a = Matrix([[5/12, -1/12],
                    [3/4, 1/4]])
        b = Vector([3/4, 1/4])
        c = Vector([1/3, 1])

        self.solver = BaseRungeImplicitODESolver(a, b, c) 

    def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(f, t_span, y0, h)
    

class LobattoIIIAODESolver(BaseODESolver):
    """
    Решение методом Лобатто IIIA (4й порядок, А-устойчив)
    """
    
    def __init__(self):
        sqrt15 = np.sqrt(15)
        a = Matrix([[0, 0, 0],
                    [5/24, 1/3, -1/24],
                    [1/6, 2/3, 1/6]])
        b = Vector([1/6, 2/3, 1/6])
        c = Vector([0, 1/2, 1])

        self.solver = BaseRungeImplicitODESolver(a, b, c) 

    def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(f, t_span, y0, h)
