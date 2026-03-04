from comp_math.ode.explicit.multistep.base_adams_solver import BaseAdamsBashforthSolver
from comp_math.ode.explicit.single_step.euler_ode_solver import EulerODESolver
from comp_math.ode.explicit.single_step.rk4_ode_solver import Rk4ODESolver


class AdamsBashforth1Solver(BaseAdamsBashforthSolver):
    """Метод Адамса-Башфорта 1 порядка (он же явный Эйлер)"""
    
    def __init__(self):
        super().__init__()
        self.order = 1
        self.coeffs = [1.0]
        self.starter = EulerODESolver()


class AdamsBashforth2Solver(BaseAdamsBashforthSolver):
    """Метод Адамса-Башфорта 2 порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 2
        self.coeffs = [1.5, -0.5]
        self.starter = Rk4ODESolver()


class AdamsBashforth3Solver(BaseAdamsBashforthSolver):
    """Метод Адамса-Башфорта 3 порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 3
        self.coeffs = [23/12, -16/12, 5/12]
        self.starter = Rk4ODESolver()


class AdamsBashforth4Solver(BaseAdamsBashforthSolver):
    """Метод Адамса-Башфорта 4 порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 4
        self.coeffs = [55/24, -59/24, 37/24, -9/24]
        self.starter = Rk4ODESolver()