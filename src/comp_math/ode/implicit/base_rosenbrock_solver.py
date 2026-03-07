import numpy as np
from typing import Callable, Optional, Tuple, List
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.objects.matrix import Matrix
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
    
    def _numerical_jacobian(self, f: Callable, t: float, y: Vector) -> np.ndarray:
        """Численное дифференцирование для получения матрицы Якоби"""
        def f_numpy(y_np: np.ndarray) -> np.ndarray:
            y_vec = Vector(y_np.tolist())
            return f(t, y_vec).to_numpy()
        
        y_np = y.to_numpy()
        J_np = NumericalJacobian.differentiate(f_numpy, y_np, 1e-8)
        return J_np
    
    def solve(self, f: Callable, dim: int, t_span: Tuple[float, float], 
              y0: List[float], h: float, 
              jacobian: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Решение системы ОДУ методом Розенброка"""
        
        t = np.arange(t_span[0], t_span[1] + h, h)
        y = np.empty(len(t), dtype=object)
        
        y0_vec = y0 if isinstance(y0, Vector) else Vector(y0)
        y[0] = y0_vec
        
        I = np.eye(dim)
        n_stages = len(self.b)
                
        for n in range(len(t) - 1):
            tn = t[n]
            yn = y[n]
            
            if jacobian:
                J_matrix = jacobian(tn, yn)
                J = J_matrix.to_numpy() if isinstance(J_matrix, Matrix) else np.array(J_matrix)
            else:
                J = self._numerical_jacobian(f, tn, yn)
            
            M = I - h * self.gamma * J
            k = []
            
            for i in range(n_stages):
                y_arg = yn
                for j in range(i):
                    y_arg = y_arg + self.a[i][j] * k[j]
                
                t_stage = tn + self.alpha[i] * h
                f_val = f(t_stage, y_arg)
                
                rhs = h * f_val.to_numpy().copy()
                sum_gamma_k = np.zeros(dim)
                for j in range(i):
                    sum_gamma_k += self.gamma_matrix[i][j] * k[j].to_numpy()
                rhs += h * J @ sum_gamma_k
                
                try:
                    k_i_np = np.linalg.solve(M, rhs)
                except np.linalg.LinAlgError:
                    k_i_np = np.linalg.lstsq(M, rhs, rcond=None)[0]
                
                k.append(Vector(k_i_np.tolist()))
            
            y_new = yn
            for i in range(n_stages):
                y_new = y_new + self.b[i] * k[i]
            
            y[n+1] = y_new
            
            if n % 1000 == 0:
                y_np = y_new.to_numpy()
                print(f"Шаг {n}: t={tn:.2e}, y=[{y_np[0]:.6f}, {y_np[1]:.6e}, {y_np[2]:.6f}]")
        
        return t, y