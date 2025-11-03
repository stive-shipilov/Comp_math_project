from typing import Callable, Tuple
from ...base_nonlinear_solver import NonlinearSolver1D


class FixedPointSolver1D(NonlinearSolver1D):
    """Метод простой итерации для 1D уравнений"""
    
    def __init__(self, lambda_param: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_param = lambda_param

    def _solve_implementation_1d(self, f: Callable[[float], float],
                                interval: Tuple[float, float],
                                x_initial: float = None) -> float:
        a, b = interval
        x = x_initial
        if x_initial is None:
            x = (a + b) / 2

        def phi(x):
            return x - self.lambda_param * f(x)

        x_prev = None
        for _ in range(self.max_iterations):
            x_new = phi(x)
            
            if x_new < a or x_new > b:
                x_new = (a + b) / 2
            
            error = abs(x_new - x)
            residual = abs(f(x_new))
            self._add_iteration(error)
            
            if error < self.tolerance and residual < self.tolerance:
                return x_new
                
            x_prev = x
            x = x_new
    
        if abs(f(x)) > self.tolerance:
            x_refined = x - f(x) * (x - x_prev) / (f(x) - f(x_prev) + 1e-15)
            if abs(f(x_refined)) < abs(f(x)):
                return x_refined
                
        return x