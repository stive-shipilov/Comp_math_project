import numpy as np
from typing import Callable, Optional, Tuple, List
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry
from comp_math.ode.base_ode_solver import BaseODESolver
from comp_math.nonlinear.nonlinear_solvers_registry import NewtonSolverND
from comp_math.differentiation.numerical.numericalJacobian import NumericalJacobian


class BaseGearSolver(BaseODESolver):
    """Базовый класс для методов Гира (BDF)"""
    
    def __init__(self):
        super().__init__()
        self.order = None
        self.alpha = None
        self.beta = None
        self.start_methods = []
        self.newton_solver = NewtonSolverND(max_iterations=50, tolerance=1e-10)
        
    def _create_bdf_function(self, f: Callable, tn: float, h: float, 
                            y_pred: Vector) -> Callable:
        """Создает функцию для метода Ньютона"""
        def G(y_np: np.ndarray) -> np.ndarray:
            y_vec = Vector(y_np.tolist())
            f_val = f(tn + h, y_vec)
            return (y_vec - y_pred - self.beta * h * f_val).to_numpy()
        return G
    
    def _compute_prediction(self, y_prev: List[Vector]) -> Vector:
        """Вычисляет предсказание"""
        dim = y_prev[0].dim
        y_pred = Vector(np.zeros(dim))
        for j in range(min(len(y_prev), len(self.alpha))):
            y_pred = y_pred + self.alpha[j] * y_prev[j]
        return y_pred
    
    def _get_jacobian(self, f: Callable, t: float, y: Vector) -> np.ndarray:
        """Возвращает якобиан (аналитический или численный)"""
        def f_numpy(y_np: np.ndarray) -> np.ndarray:
            return f(t, Vector(y_np.tolist())).to_numpy()
        return NumericalJacobian.differentiate(f_numpy, y.to_numpy(), 1e-8)
    
    def solve(self, f: Callable, dim: int, t_span: Tuple[float, float], 
              y0: List[float], h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Решает систему ОДУ методом Гира (BDF)"""
        t = [t_span[0]]
        y = [y0 if isinstance(y0, Vector) else Vector(y0)]
        
        current_h = min(h, 1e-4)
        t_current = t_span[0]
        
        step = 0
        while t_current < t_span[1]:
            step += 1
            if t_current + current_h > t_span[1]:
                current_h = t_span[1] - t_current
            
            J = self._get_jacobian(f, t_current, y[-1])
            
            if len(y) <= self.order:
                # Разгон
                M = np.eye(dim) - current_h * J
                f_val = f(t_current + current_h, y[-1])
                rhs = y[-1].to_numpy() + current_h * f_val.to_numpy()
                cg_solver = SLASolverRegistry.create_solver("cg")
                y_new_np = cg_solver.solve(Matrix(M), Vector(rhs)).to_numpy()
                y_new = Vector(y_new_np.tolist())
            else:
                # Основной метод Гира
                y_prev = [y[-1 - j] for j in range(min(self.order, len(y)))]
                y_pred = self._compute_prediction(y_prev)
                G = self._create_bdf_function(f, t_current, current_h, y_pred)
                
                def J_func(y_np: np.ndarray) -> np.ndarray:
                    return np.eye(dim) - self.beta * current_h * J
                
                x0 = y_pred.to_numpy()
                y_new_np = self.newton_solver.solve(F=G, x0=x0, J=J_func)
                y_new = Vector(y_new_np.tolist())
            
            t.append(t_current + current_h)
            y.append(y_new)
            t_current += current_h
            
        return np.array(t), np.array(y, dtype=object)
