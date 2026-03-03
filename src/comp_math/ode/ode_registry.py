from typing import Dict, Type
from ..core.base_method_registry import BaseMethodRegistry
from .base_ode_solver import BaseODESolver
from .explicit.euler_ode_solver import EulerODESolver
from .explicit.heun_ode_solver import HeunODESolver
from .explicit.kutta_third_ode_solver import KuttaThirdODESolver
from .explicit.rk4_ode_solver import Rk4ODESolver

class ODERegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров решателей ОДУ"""
    
    _solvers: Dict[str, Type[BaseODESolver]] = {
        "euler": EulerODESolver,
        "kutta": KuttaThirdODESolver,
        "heun": HeunODESolver,
        "rk4": Rk4ODESolver
    }
    