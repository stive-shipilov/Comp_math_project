from typing import List, Tuple, Callable, Optional
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.nonlinear.nonlinear_solvers_registry import NewtonSolverND
from comp_math.differentiation.numerical.numericalJacobian import NumericalJacobian
from comp_math.ode.base_ode_solver import BaseODESolver
from comp_math.ode.explicit.single_step.rk4_ode_solver import Rk4ODESolver
import numpy as np


class BaseBDFSolver(BaseODESolver):
    """Базовый класс для методов ФДН"""
    
    def __init__(self):
        super().__init__()
        self.order = None
        self.alpha = None
        self.beta = None
        self.y_history = []
        self.f_history = []
        self.t_history = []
        
        self.starter = Rk4ODESolver()
        
        self.newton_solver = NewtonSolverND(max_iterations=10, tolerance=1e-10)
    
    def _startup(self, f: Callable, dim: int, t0: float, y0: Vector, h: float):
        """Разгон метода"""
        t_vals = [t0]
        y_vals = [y0]
        f_vals = [f(t0, y0)]
        
        if self.order == 1:
            return t_vals, y_vals, f_vals
        
        current_t = t0
        current_y = y0
        
        for _ in range(self.order - 1):
            t_next, y_next = self.starter.solve(f, dim, (current_t, current_t + h), 
                                                current_y.to_numpy(), h)
            current_t = t_next[-1]
            current_y = y_next[-1]
            t_vals.append(current_t)
            y_vals.append(current_y)
            f_vals.append(f(current_t, current_y))
        
        return t_vals, y_vals, f_vals
    
    def _create_bdf_function(self, f: Callable, t_next: float, rhs: Vector, h: float, dim: int):
        """Создает функцию для метода Ньютона вида"""
        def G(y_np: np.ndarray) -> np.ndarray:
            y_vec = Vector(y_np.tolist())
        
            f_val = f(t_next, y_vec)
            result_vec = y_vec - (h * self.beta) * f_val - rhs
            
            return result_vec.to_numpy()
        
        return G
    
    def _create_jacobian_function(self, f: Callable, t_next: float, h: float):
        """Создает функцию для вычисления якобиана системы"""
        def J(y_np: np.ndarray) -> np.ndarray:
            y_vec = Vector(y_np.tolist())
            
            # Создаем функцию для NumericalJacobian
            def F_for_jacobian(y: np.ndarray) -> np.ndarray:
                y_vec = Vector(y.tolist())
                return f(t_next, y_vec).to_numpy()
            
            # Вычисляем якобиан от f
            Jf = NumericalJacobian.differentiate(F_for_jacobian, y_np, 1e-6)
            
            # Якобиан системы: I - h*beta*Jf
            dim = len(y_np)
            I = np.eye(dim)
            return I - h * self.beta * Jf
        
        return J
    
    def solve(self, f: Callable, dim: int, t_span: Tuple[float, float], 
          y0: List[float], h: float, 
          jacobian: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Решение ОДУ методом ФДН"""
        
        if self.alpha is None:
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
        
        # Разгон
        t_start, y_start, f_start = self._startup(f, dim, t[0], y0_vec, h)
        for i in range(len(t_start)):
            t[i] = t_start[i]
            y[i] = y_start[i]
            self.t_history.append(t_start[i])
            self.y_history.append(y_start[i])
            self.f_history.append(f_start[i])
        
        current_idx = len(t_start) - 1
        
        while current_idx < n_steps - 1:
            # Правая часть из истории
            rhs = Vector([0.0] * dim)
            for j in range(1, len(self.alpha)):
                rhs += -self.alpha[j] * self.y_history[-j]
            
            # Начальное приближение
            y_guess = (self.y_history[-1] * 2 - self.y_history[-2]) if len(self.y_history) >= 2 else self.y_history[-1]
            t_next = self.t_history[-1] + h
            
            # Функция для Ньютона
            G = self._create_bdf_function(f, t_next, rhs, h, dim)
            J = jacobian if jacobian else self._create_jacobian_function(f, t_next, h)
            
            # Решаем Ньютоном
            try:
                y_next = Vector(self.newton_solver.solve(F=G, x0=y_guess.to_numpy(), J=J).tolist())
            except:
                y_next = Vector(self.newton_solver.solve(F=G, x0=self.y_history[-1].to_numpy(), J=J).tolist())
            
            current_idx += 1
            t[current_idx] = t_next
            y[current_idx] = y_next
            
            self.t_history.append(t_next)
            self.y_history.append(y_next)
            self.f_history.append(f(t_next, y_next))
            
            # Ограничиваем историю
            if len(self.t_history) > self.order:
                self.t_history.pop(0)
                self.y_history.pop(0)
                self.f_history.pop(0)
        
        return t, y
