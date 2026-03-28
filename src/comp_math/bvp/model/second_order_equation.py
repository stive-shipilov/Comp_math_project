import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import root

from comp_math.bvp.model.bvp_problem import BVProblem
from comp_math.bvp.model.first_order_system import FirstOrderSystem


class SecondOrderEquation(BVProblem):
    """
    Одно уравнение второго порядка:
        y'' = f(x, y, y')
        y(a) = alpha, y(b) = beta
    
    Пример:
        def f(x, y, yp):
            return -y
        
        problem = SecondOrderEquation(f, alpha=1, beta=0, domain=(0, 1))
    """
    
    def __init__(self, f: Callable, alpha: float, beta: float, domain: Tuple[float, float]):
        super().__init__(domain)
        self._f = f
        self._alpha = alpha
        self._beta = beta
    
    def get_type(self) -> str:
        return "second_order_equation"
    
    def get_f(self) -> Callable:
        return self._f
    
    def get_alpha(self) -> float:
        return self._alpha
    
    def get_beta(self) -> float:
        return self._beta
    
    def to_first_order_system(self) -> FirstOrderSystem:
        """Преобразует в систему первого порядка для стрельбы"""
        def system(x, y):
            return np.array([y[1], self._f(x, y[0], y[1])])
        
        def bc(ya, yb):
            return np.array([ya[0] - self._alpha, yb[0] - self._beta])
        
        return FirstOrderSystem(system, bc, self.domain)