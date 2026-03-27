import numpy as np
from scipy.integrate import solve_bvp
from dataclasses import dataclass
from typing import List, Tuple, Union, Callable, Dict
import matplotlib.pyplot as plt
import inspect


class Variable:
    def __init__(self, name: str):
        self.name = name
        self._index = None
    
    def at(self, point: float):
        return BoundaryCondition(self, point, derivative_order=0)
    
    def __repr__(self):
        return self.name


class Derivative:
    def __init__(self, var: Variable, order: int):
        self.var = var
        self.order = order
    
    def __eq__(self, other):
        return Equation(self, other)
    
    def at(self, point: float):
        return BoundaryCondition(self.var, point, derivative_order=self.order)


class BoundaryCondition:
    def __init__(self, var: Variable, point: float, derivative_order: int = 0):
        self.var = var
        self.point = point
        self.derivative_order = derivative_order
    
    def __eq__(self, other):
        return BoundaryEquation(self, other)


class Equation:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class BoundaryEquation:
    def __init__(self, bc, value):
        self.bc = bc
        self.value = value