from typing import List, Tuple, Callable
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.ode.explicit.single_step.euler_ode_solver import EulerODESolver
from comp_math.ode.explicit.single_step.rk4_ode_solver import Rk4ODESolver
from ...base_ode_solver import BaseODESolver
import numpy as np


class BaseAdamsBashforthSolver(BaseODESolver):
    """Базовый класс для всех методов Адамса-Башфорта"""
    
    def __init__(self):
        super().__init__()
        self.coeffs = None
        self.order = None
        self.starter = None
        self.y_history = []
        self.f_history = []
        self.t_history = []
    
    def _get_starter(self):
        """Возвращает подходящий стартер для данного порядка"""
        if self.order == 1:
            return EulerODESolver()
        else:
            return Rk4ODESolver()
    
    def _startup(self, f: Callable, dim: int, t0: float, y0: Vector, h: float):
        """Разгонка методов"""
        t_vals = [t0]
        y_vals = [y0]
        f_vals = [f(t0, y0)]
        
        if self.order == 1:
            return t_vals, y_vals, f_vals
        
        starter = self._get_starter()
        current_t = t0
        current_y = y0
        
        for _ in range(self.order - 1):
            t_next, y_next = starter.solve(f, dim, (current_t, current_t + h), current_y.to_numpy(), h)
            current_t = t_next[-1]
            current_y = y_next[-1]
            t_vals.append(current_t)
            y_vals.append(current_y)
            f_vals.append(f(current_t, current_y))
        
        return t_vals, y_vals, f_vals
    
    def solve(self, f: Callable, dim: int, t_span: Tuple[float, float], y0: List[float], h: float) \
        -> Tuple[np.ndarray, np.ndarray]:
        
        if self.coeffs is None:
            raise ValueError("Коэффициенты не заданы")
        
        n_steps = int((t_span[1] - t_span[0]) / h) + 1
        t = np.empty(n_steps, dtype=float)
        y = np.empty(n_steps, dtype=object)
        
        t[0] = t_span[0]
        y0_vec = y0 if isinstance(y0, Vector) else Vector(y0)
        y[0] = y0_vec
        
        self.y_history = []
        self.f_history = []
        self.t_history = []
        
        t_start, y_start, f_start = self._startup(f, dim, t[0], y0_vec, h)
        
        for i in range(len(t_start)):
            if i < n_steps:
                t[i] = t_start[i]
                y[i] = y_start[i]
                self.t_history.append(t_start[i])
                self.y_history.append(y_start[i])
                self.f_history.append(f_start[i])
        
        current_idx = len(t_start) - 1
        
        while current_idx < n_steps - 1:
            recent_f = self.f_history[-self.order:]
            y_n = self.y_history[-1]
            
            sum_f = Vector([0.0] * dim)
            for j, beta in enumerate(self.coeffs):
                sum_f += beta * recent_f[-1 - j]
            
            y_next = y_n + h * sum_f
            
            current_idx += 1
            t_current = self.t_history[-1] + h
            t[current_idx] = t_current
            y[current_idx] = y_next
            
            self.t_history.append(t_current)
            self.y_history.append(y_next)
            self.f_history.append(f(t_current, y_next))
            
            if len(self.t_history) > self.order:
                self.t_history.pop(0)
                self.y_history.pop(0)
                self.f_history.pop(0)
        
        return t, y
