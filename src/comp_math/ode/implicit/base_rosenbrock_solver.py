import numpy as np
from typing import Callable, Optional, Tuple, List
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry
from comp_math.ode.base_ode_solver import BaseODESolver
from comp_math.differentiation.numerical.numericalJacobian import NumericalJacobian


class BaseRosenbrockSolver(BaseODESolver):
    """Базовый класс для методов Розенброка"""
    
    def __init__(self):
        super().__init__()
        self.gamma = None
        self.a = None
        self.b = None
        self.alpha = None
        self.gamma_matrix = None
        self.order = None
    
    def _numerical_jacobian(self, f: Callable, t: float, y: Vector) -> Matrix:
        """Численное дифференцирование для получения матрицы Якоби"""
        def f_numpy(y_np: np.ndarray) -> np.ndarray:
            y_vec = Vector(y_np.tolist())
            return f(t, y_vec).to_numpy()
        
        y_np = y.to_numpy()
        J_np = NumericalJacobian.differentiate(f_numpy, y_np, 1e-8)
        return Matrix(J_np.tolist())
    
    def solve(self, f: Callable, dim: int, t_span: Tuple[float, float], 
              y0: List[float], h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Решение системы ОДУ методом Розенброка"""
        
        t = np.arange(t_span[0], t_span[1] + h, h)
        y = np.empty(len(t), dtype=object)
        
        y0_vec = y0 if isinstance(y0, Vector) else Vector(y0)
        y[0] = y0_vec
        
        I = Matrix(np.eye(dim))
        n_stages = len(self.b)
                
        for n in range(len(t) - 1):
            tn = t[n]
            yn = y[n]
            
            J = self._numerical_jacobian(f, tn, yn)
            
            M = I.subtract(J * (h * self.gamma))
            k = []
            
            for i in range(n_stages):
                y_arg = yn
                for j in range(i):
                    y_arg = y_arg + k[j] * self.a[i][j]
                
                t_stage = tn + self.alpha[i] * h
                f_val = f(t_stage, y_arg)
                
                rhs = f_val * h
                sum_gamma_k = Vector(np.zeros(dim))
                for j in range(i):
                    sum_gamma_k = sum_gamma_k + k[j] * self.gamma_matrix[i][j]
                rhs = rhs + (J * sum_gamma_k) * h
                
                bcg_solver = SLASolverRegistry.create_solver("zeidel")
                k_i = bcg_solver.solve(M, rhs)
                k.append(k_i)
            
            y_new = yn
            for i in range(n_stages):
                y_new = y_new + k[i] * self.b[i]
            
            y[n+1] = y_new
        
        return t, y