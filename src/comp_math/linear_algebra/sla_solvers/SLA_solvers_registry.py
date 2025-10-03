from typing import Dict, Type, Any
from .solvers.direct.gauss_SLA_solver import GaussSolver
from .solvers.iterative.jacobi_SLA_solver import JacobiSolver
from .solvers.iterative.zeidel_SLA_solver import ZeidelSolver
from .solvers.iterative.relaxation_SLA_solver import RelaxationSolver
from .solvers.variational.cg_SLA_solver import CGsolver
from .solvers.variational.bcg_SLA_solver import BCGsolver
from .solvers.variational.sbcg_SLA_solver import SBCGsolver
from .base_SLA_solver import SLASolver


class SLASolverRegistry:
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

    @classmethod
    def create_solver(cls, method: str, **kwargs: Any) -> SLASolver:
        """Создает экземпляр решателя по имени метода"""
        if method not in cls._solvers:
            raise ValueError(f"Неизвестный метод '{method}'. Доступны только: {cls._solvers.keys()}")
        
        solver_class = cls._solvers[method]
        return solver_class(**kwargs)

    @classmethod
    def get_available_solvers(cls) -> list:
        """Возвращает список доступных методов"""
        return list(cls._solvers.keys())

