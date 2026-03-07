from typing import Dict, Type

from comp_math.ode.explicit.multistep.adams_impl import (AdamsBashforth1Solver, AdamsBashforth2Solver, 
AdamsBashforth3Solver, AdamsBashforth4Solver)
from comp_math.ode.implicit.rosenbrock_impl import Rosenbrock1Solver, Rosenbrock2Solver, Rosenbrock4Solver
from comp_math.ode.implicit.gear_solver_impl import Gear1Solver, Gear2Solver, Gear3Solver, Gear4Solver
# from comp_math.ode.implicit.bdf_solver_impl import (BDF1Solver, BDF2Solver, BDF3Solver, BDF4Solver)
from ..core.base_method_registry import BaseMethodRegistry
from .base_ode_solver import BaseODESolver
from .explicit.single_step.euler_ode_solver import EulerODESolver
from .explicit.single_step.heun_ode_solver import HeunODESolver
from .explicit.single_step.kutta_third_ode_solver import KuttaThirdODESolver
from .explicit.single_step.rk4_ode_solver import Rk4ODESolver
from .implicit.runge_solver_impl import (GaussLegendre2ODESolver, GaussLegendre4ODESolver, RadoIIAODESolver, LobattoIIIAODESolver)

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
        "adam4": AdamsBashforth4Solver,
        "gear1": Gear1Solver,
        "gear2": Gear2Solver,
        "gear3": Gear3Solver,
        "gear4": Gear4Solver,
        "gauss_legendre_2": GaussLegendre2ODESolver,
        "gauss_legendre_4": GaussLegendre4ODESolver,
        "rado": RadoIIAODESolver,
        "lobatto": LobattoIIIAODESolver,
        "rosenbrock1": Rosenbrock1Solver,
        "rosenbrock2": Rosenbrock2Solver,
        "rosenbrock4": Rosenbrock4Solver
}
    