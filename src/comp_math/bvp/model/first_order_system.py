import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import root

from comp_math.bvp.model.bvp_problem import BVProblem


class FirstOrderSystem(BVProblem):
    """
    Система первого порядка:
        y' = f(x, y)
        g(ya, yb) = 0
    
    Пример как её задвать:
        def system(x, y):
            return np.array([y[1], -y[0]])
        
        def bc(ya, yb):
            return np.array([ya[0] - 1, yb[0]])
        
        problem = FirstOrderSystem(system, bc, domain=(0, 1))
    """
    
    def __init__(self, system: Callable, bc: Callable, domain: Tuple[float, float]):
        super().__init__(domain)
        self._system = system
        self._bc = bc
        self._n_vars = len(system(domain[0], np.zeros(2)))
    
    def get_type(self) -> str:
        return "first_order_system"
    
    def get_system(self) -> Callable:
        return self._system
    
    def get_bc(self) -> Callable:
        return self._bc
    
    def get_n_vars(self) -> int:
        return self._n_vars

