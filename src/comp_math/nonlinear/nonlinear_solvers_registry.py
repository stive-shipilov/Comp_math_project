from typing import Dict, Type, Any
from .base_nonlinear_solver import NonlinearSolver
from .solvers.iterative.bisection_nonlinear import BisectionSolver
from .solvers.iterative.newton_solver import NewtonSolver1D, NewtonSolverND
from .solvers.iterative.fixed_points_nonlinear import FixedPointSolver1D
from .solvers.variation.variational_nonlinear import VariationalEquationSolver1D
from ..core.base_method_registry import BaseMethodRegistry


class NonlinearSolverRegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров решателей СЛАУ"""
    
    _solvers: Dict[str, Type[NonlinearSolver]] = {
        "bisection": BisectionSolver,
        "newton1D": NewtonSolver1D,
        "newtonND": NewtonSolverND,
        "fixedPoints1D": FixedPointSolver1D, 
        "variational": VariationalEquationSolver1D
    }
