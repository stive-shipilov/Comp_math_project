from typing import Dict, Type, Any

from .solvers.direct.gauss_SLA_solver import GaussSolver
from .solvers.iterative.jacobi_SLA_solver import JacobiSolver
from .solvers.iterative.zeidel_SLA_solver import ZeidelSolver
from .solvers.iterative.relaxation_SLA_solver import RelaxationSolver
from .solvers.variational.cg_SLA_solver import CGsolver
from .solvers.variational.bcg_SLA_solver import BCGsolver
from .solvers.variational.sbcg_SLA_solver import SBCGsolver
from .base_SLA_solver import SLASolver
from comp_math.core.base_method_registry import BaseMethodRegistry


class SLASolverRegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров решателей СЛАУ"""
    
    _solvers: Dict[str, Type[SLASolver]] = {
        "gauss": GaussSolver,
        "jacobi": JacobiSolver,
        "zeidel": ZeidelSolver,
        "relaxation": RelaxationSolver,
        "cg": CGsolver,
        "bcg": BCGsolver,
        "sbcg": SBCGsolver
    }
