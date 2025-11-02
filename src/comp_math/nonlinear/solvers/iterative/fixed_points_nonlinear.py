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

        for _ in range(self.max_iterations):
            x_new = phi(x)
            
            # Чтобы не выходило за свой интервал локализации
            if x_new < a or x_new > b:
                x_new = (a + b) / 2
            
            error = abs(x_new - x)
            self._add_iteration(error)
            
            if error < self.tolerance or abs(f(x_new)) < self.tolerance:
                return x_new
                
            x = x_new
    
        return x