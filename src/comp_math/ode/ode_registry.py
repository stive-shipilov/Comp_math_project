from typing import Dict, Type

from comp_math.ode.explicit.multistep.adams_impl import (AdamsBashforth1Solver, AdamsBashforth2Solver, 
AdamsBashforth3Solver, AdamsBashforth4Solver)
from ..core.base_method_registry import BaseMethodRegistry
from .base_ode_solver import BaseODESolver
from .explicit.single_step.euler_ode_solver import EulerODESolver
from .explicit.single_step.heun_ode_solver import HeunODESolver
from .explicit.single_step.kutta_third_ode_solver import KuttaThirdODESolver
from .explicit.single_step.rk4_ode_solver import Rk4ODESolver

class ODERegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров решателей ОДУ"""
    
    _solvers: Dict[str, Type[BaseODESolver]] = {
        "euler": EulerODESolver,
        "kutta": KuttaThirdODESolver,
        "heun": HeunODESolver,
        "rk4": Rk4ODESolver,
        "adam1": AdamsBashforth1Solver,
        "adam2": AdamsBashforth2Solver,
        "adam3": AdamsBashforth3Solver,
        "adam4": AdamsBashforth4Solver
}
    