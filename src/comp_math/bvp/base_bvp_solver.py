import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import root

from comp_math.bvp.model.bvp_problem import BVProblem
from comp_math.core.base_solver import BaseNumericalMethod


class BaseBVPSolver(BaseNumericalMethod):
    """Абстрактный базовый класс для солверов"""
    
    def __init__(self, problem: BVProblem):
        self.problem = problem
        self.a, self.b = problem.get_domain()
        self._check_compatibility()
    
    def _check_compatibility(self):
        """Проверяет, подходит ли задача для этого солвера"""
        expected_type = self._get_expected_type()
        actual_type = self.problem.get_type()
        
        if actual_type != expected_type:
            raise TypeError(
                f"{self.__class__.__name__} требует {expected_type}, "
                f"получен {actual_type}"
            )
    
    @abstractmethod
    def _get_expected_type(self) -> str:
        """Возвращает ожидаемый тип задачи"""
        pass
    
    @abstractmethod
    def solve(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Решает задачу, возвращает (x, y)"""
        pass