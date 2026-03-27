import numpy as np
from scipy.integrate import solve_bvp
from dataclasses import dataclass
from typing import List, Tuple, Union, Callable, Dict
import matplotlib.pyplot as plt
import inspect

from comp_math.bvp.model.bvp_problem import BVProblem
from comp_math.bvp.model.bvp_result import BVPResult
from comp_math.bvp.model.model_bvp import BoundaryEquation, Derivative, Equation, Variable


class BVPResult:
    def __init__(self, sol, original_vars, system_vars):
        self.sol = sol
        self.original_vars = original_vars
        self.system_vars = system_vars
    
    def get(self, var):
        if isinstance(var, str):
            for v in self.original_vars:
                if v.name == var:
                    var = v
                    break
        if isinstance(var, Derivative):
            target = f"{var.var.name}_{var.order}"
            for sv in self.system_vars:
                if sv.name == target:
                    return lambda x: self.sol.sol(x)[sv._index]
        if isinstance(var, Variable):
            for sv in self.system_vars:
                if sv.name == var.name:
                    return lambda x: self.sol.sol(x)[sv._index]
        raise ValueError(f"Переменная {var} не найдена")
    
    def plot(self, var, x_range=None):
        if x_range is None:
            x_range = (self.sol.x[0], self.sol.x[-1])
        x = np.linspace(x_range[0], x_range[1], 200)
        y = self.get(var)(x)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, linewidth=2)
        plt.xlabel('x')
        plt.ylabel(str(var))
        plt.grid(True, alpha=0.3)
        plt.show()
