from typing import Dict, Type, Any
from .base_nonlinear_solver import NonlinearSolver
from .solvers.iterative.bisection_nonlinear import BisectionSolver
from .solvers.iterative.newton_solver import NewtonSolver1D, NewtonSolverND
from .solvers.iterative.fixed_points_nonlinear import FixedPointSolver1D
from .solvers.variation.variational_nonlinear import VariationalEquationSolver1D

class NonlinearSolverRegistry:
    """Фабрика для создания экземпляров решателей СЛАУ"""
    
    _solvers: Dict[str, Type[NonlinearSolver]] = {
        "bisection": BisectionSolver,
        "newton1D": NewtonSolver1D,
        "newtonND": NewtonSolverND,
        "fixedPoints1D": FixedPointSolver1D, 
        "variational": VariationalEquationSolver1D
    }

    @classmethod
    def create_solver(cls, method: str, **kwargs: Any) -> NonlinearSolver:
        """Создает экземпляр решателя по имени метода"""
        if method not in cls._solvers:
            raise ValueError(f"Неизвестный метод '{method}'. Доступны только: {cls._solvers.keys()}")
        
        solver_class = cls._solvers[method]
        return solver_class(**kwargs)

    @classmethod
    def get_available_solvers(cls) -> list:
        """Возвращает список доступных методов"""
        return list(cls._solvers.keys())

